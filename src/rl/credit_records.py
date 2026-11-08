"""Append-only S1 raw credit and S2 heldout hooks, from the first actual update.

No variance, world-component decomposition, success-rate aggregate, or AUC is
computed here. These raw observations permit later independently reviewed
statistics without choosing ddof, world weighting, or integration endpoints.
The trainer adapter must call begin_update BEFORE optimization and finish_update
after one full rollout-batch PPO actor/critic cycle, not configured step IDs or
individual minibatch AdamW calls. This matches the existing completed-step logs.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Mapping

from .credit import ARMS, PAIRED_ARMS, CreditContractError, advantages, prior, q_commit, same_world_credit


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class TrainingHoldout:
    manifest_path: str
    manifest_sha256: str
    episodes: tuple[tuple[str, str, str, str, int], ...]  # eval id, pair id, scene, world, seed
    policy_contract: str
    partition_proof: Mapping

    @classmethod
    def load(cls, path, *, train_scenes, frozen_test_scenes, optimization_pair_ids, frozen_test_pair_ids):
        train_scenes, frozen_test_scenes = set(train_scenes), set(frozen_test_scenes)
        optimization_pair_ids, frozen_test_pair_ids = set(optimization_pair_ids), set(frozen_test_pair_ids)
        path = Path(path).resolve()
        data = json.loads(path.read_text())
        if (data.get("role") != "train_scene_holdout" or data.get("frozen_before_training") is not True
                or not data.get("evaluation_policy_contract") or not data.get("episodes")):
            raise CreditContractError("S2 requires an explicitly frozen training-scene holdout manifest")
        if set(train_scenes) & set(frozen_test_scenes):
            raise CreditContractError("Train and frozen test scene declarations overlap")
        rows, seen = [], set()
        for item in data["episodes"]:
            required = ("eval_episode_id", "pair_id", "scene", "world_id", "seed")
            if any(key not in item for key in required):
                raise CreditContractError("Incomplete heldout episode identity")
            if (item["eval_episode_id"] in seen or item["scene"] not in train_scenes
                    or item["scene"] in frozen_test_scenes or item["world_id"] not in ("w0", "w1")
                    or item["pair_id"] in optimization_pair_ids or item["pair_id"] in frozen_test_pair_ids
                    or type(item["seed"]) is not int):
                raise CreditContractError("S2 leakage, repeated episode, or invalid manifest identity")
            seen.add(item["eval_episode_id"])
            rows.append(tuple(item[key] for key in required))
        proof = {"declared_train_scenes": sorted(train_scenes), "excluded_frozen_test_scenes": sorted(frozen_test_scenes),
                 "optimization_pair_ids_sha256": digest(sorted(optimization_pair_ids)),
                 "frozen_test_pair_ids_sha256": digest(sorted(frozen_test_pair_ids)),
                 "caller_must_source_partition_sets_from_frozen_manifests": True}
        return cls(str(path), file_digest(path), tuple(rows), data["evaluation_policy_contract"], proof)

    def verify_unchanged(self):
        if file_digest(self.manifest_path) != self.manifest_sha256:
            raise CreditContractError("The pretraining S2 holdout manifest changed")


class RawTrainingRecorder:
    """One writer/learner; reject overwrites and concurrent or ambiguous updates."""
    def __init__(self, directory, *, run_id, arm, seed, rho, common_config_sha256,
                 holdout: TrainingHoldout, max_decisions=8, chunk_bytes=32 * 1024 * 1024, allow_synthetic=False):
        if (arm not in ARMS or not run_id or type(seed) is not int
                or not re.fullmatch(r"[0-9a-f]{64}", common_config_sha256)
                or type(max_decisions) is not int or max_decisions <= 0
                or type(chunk_bytes) is not int or chunk_bytes <= 0):
            raise CreditContractError("Require explicit training identity, frozen common-config hash, and valid bounds")
        prior(rho)
        holdout.verify_unchanged()
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.identity = {"run_id": run_id, "arm": arm, "seed": seed, "rho": rho,
                         "common_config_sha256": common_config_sha256, "max_decisions": max_decisions,
                         "synthetic_fixture_session": allow_synthetic,
                         "completed_update_unit": "one_full_rollout_batch_PPO_critic_actor_cycle"}
        self.holdout, self.chunk_bytes = holdout, chunk_bytes
        self.event_id, self.attempts, self.completed_updates = 0, 0, 0
        self.active, self.s2_due, self.closed = None, None, False
        self.stream, self.stream_size, self.chunks = None, 0, []
        self._append({"kind": "session", **self.identity, "holdout": asdict(holdout),
            "s1_scope": "every pre-evidence decision in every attempted batch; optimizer completion recorded separately",
            "s1_statistics": None, "s2_interval_completed_updates": 20, "s2_auc": None,
            "notes": ["Raw per-world components are retained; no ddof or between-world decomposition is chosen.",
                      "S2 starts at completed optimizer update 20; no synthetic initial/terminal endpoint is added.",
                      "Q scores retain full action probability mass and complete scoring provenance.",
                      "All journal files are bounded, append-only and indexed by SHA at close."]})

    def _append(self, row):
        if self.closed:
            raise CreditContractError("Recorder is closed")
        event = {"event_id": self.event_id + 1, **row}
        raw = (json.dumps(event, sort_keys=True, allow_nan=False) + "\n").encode()
        if len(raw) > self.chunk_bytes:
            raise CreditContractError("One raw record exceeds the journal chunk bound; do not silently truncate")
        if self.stream is None or self.stream_size + len(raw) > self.chunk_bytes:
            if self.stream is not None:
                self.stream.flush()
                os.fsync(self.stream.fileno())
                self.stream.close()
            path = self.directory / f"credit_{len(self.chunks):05d}.jsonl"
            self.stream = path.open("xb")
            self.stream_size = 0
            self.chunks.append(path)
        self.stream.write(raw)
        self.stream.flush()
        self.stream_size += len(raw)
        self.event_id += 1

    def begin_update(self, plan, steps, *, applied_advantages: Mapping,
                     advantage_transform, policy_id, episode_summaries):
        if self.active is not None or self.s2_due is not None:
            raise CreditContractError("Finish/abort the active update and record any due S2 evaluation before another update")
        if not policy_id or not advantage_transform:
            raise CreditContractError("Policy identity and explicit applied-advantage transformation are required")
        if len(plan) != 64 or len({row["episode_id"] for row in plan}) != 64:
            raise CreditContractError("Every arm requires exactly 64 distinct episode instances per update")
        by_episode = {row["episode_id"]: row for row in plan}
        pairs = {}
        for row in plan:
            if row["rho"] != self.identity["rho"]:
                raise CreditContractError("Batch rho differs from the session")
            pairs.setdefault(row["pair_instance_id"], []).append(row["world_id"])
        if self.identity["arm"] in PAIRED_ARMS:
            if len(pairs) != 32 or any(sorted(worlds) != ["w0", "w1"] for worlds in pairs.values()):
                raise CreditContractError("Paired arms require 32 complete world pairs")
        elif len(pairs) != 64 or any(len(worlds) != 1 for worlds in pairs.values()):
            raise CreditContractError("Single-world arms require 64 independent pool draws")
        if set(episode_summaries) != set(by_episode):
            raise CreditContractError("Episode-level completion/budget records must cover the complete batch")
        groups = {episode: [] for episode in by_episode}
        for step in steps:
            if step.synthetic_fixture and not self.identity["synthetic_fixture_session"]:
                raise CreditContractError("Synthetic credit fixtures cannot enter a real training journal")
            if step.scores is not None and step.scores.policy_id != policy_id:
                raise CreditContractError("Q_commit was scored under a different policy snapshot")
            if step.episode_id not in by_episode:
                raise CreditContractError("Unplanned episode in credit records")
            row = by_episode[step.episode_id]
            if (step.pair_id != row["pair_id"] or step.pair_instance_id != row["pair_instance_id"]
                    or step.world_id != row["world_id"] or step.rho != row["rho"]):
                raise CreditContractError("Credit row does not match its planned pair/world/prior")
            groups[step.episode_id].append(step.decision_index)
        for indices in groups.values():
            if not 1 <= len(indices) <= self.identity["max_decisions"] or sorted(indices) != list(range(len(indices))):
                raise CreditContractError("Missing, duplicate or unbounded per-episode decision records")
        raw_advantages = advantages(self.identity["arm"], steps)
        if set(applied_advantages) != set(raw_advantages):
            raise CreditContractError("Applied advantages must cover every actual decision")
        for key, applied in applied_advantages.items():
            if type(applied) not in (int, float) or not math.isfinite(applied):
                raise CreditContractError("Applied advantage is not finite")
            if ((advantage_transform == "identity" or self.identity["arm"] in ("T0", "T1"))
                    and applied != raw_advantages[key]):
                raise CreditContractError("Identity transform/T0/T1 must preserve the supplied upstream GAE or computed scalar exactly")
        if self.identity["arm"] in ("T3", "T4"):
            shared = {}
            for step in steps:
                if step.pre_evidence:
                    shared.setdefault((step.pair_instance_id, step.decision_index), []).append(applied_advantages[step.key])
            if any(len(values) != 2 or values[0] != values[1] for values in shared.values()):
                raise CreditContractError("Applied normalization broke identical pre-evidence paired advantages")
        self.attempts += 1
        self.active = self.attempts
        all_keys = [[step.episode_id, step.decision_index] for step in steps]
        pre_keys = [[step.episode_id, step.decision_index] for step in steps if step.pre_evidence]
        self._append({"kind": "update_begin", "update_attempt": self.active,
            "completed_optimizer_updates_before": self.completed_updates, "policy_id": policy_id,
            "planned_episodes": plan, "episode_summaries": episode_summaries,
            "all_decision_keys": all_keys, "expected_pre_evidence_keys": pre_keys,
            "advantage_transform": advantage_transform})
        for step in steps:
            if not step.pre_evidence:
                continue
            q = q_commit(step.scores, step.target_name) if step.scores is not None else None
            # Even T3 retains both unaveraged worlds, enabling offline decomposition.
            component = (same_world_credit(step) if step.scores is not None and step.action_kind in ("observe", "select") else None)
            self._append({"kind": "s1_pre_evidence_decision", **self.identity,
                "update_attempt": self.active, "completed_optimizer_updates_before": self.completed_updates,
                "decision": asdict(step), "scores": step.scores.record() if step.scores is not None else None,
                "p_world": prior(step.rho)[step.world_id], "q_commit": q,
                "same_world_unaveraged_credit": component, "raw_advantage": raw_advantages[step.key],
                "applied_advantage": applied_advantages[step.key], "advantage_transform": advantage_transform})
        self._append({"kind": "s1_batch_complete", "update_attempt": self.active,
                      "pre_evidence_keys_sha256": digest(pre_keys), "pre_evidence_row_count": len(pre_keys),
                      "all_decision_row_count": len(all_keys), "episode_count": 64})
        self.stream.flush()
        os.fsync(self.stream.fileno())
        return self.active

    def finish_update(self, *, actual_completed_optimizer_updates, policy_checkpoint_id, error=None):
        if self.active is None or not policy_checkpoint_id:
            raise CreditContractError("No active update or missing policy/checkpoint identity")
        expected = self.completed_updates + (0 if error is not None else 1)
        if type(actual_completed_optimizer_updates) is not int or actual_completed_optimizer_updates != expected:
            raise CreditContractError("Use the actual contiguous optimizer completion counter, never trainer.total_training_steps")
        self._append({"kind": "update_end", "update_attempt": self.active,
            "actual_completed_optimizer_updates": actual_completed_optimizer_updates,
            "optimizer_update_completed": error is None, "error": error,
            "policy_checkpoint_id": policy_checkpoint_id})
        self.completed_updates = actual_completed_optimizer_updates
        self.active = None
        if error is None and self.completed_updates % 20 == 0:
            self.holdout.verify_unchanged()
            self.s2_due = {"actual_completed_optimizer_updates": self.completed_updates,
                           "policy_checkpoint_id": policy_checkpoint_id, "holdout": asdict(self.holdout)}
            self._append({"kind": "s2_due", **self.s2_due})
        return self.s2_due

    def record_s2(self, episodes, *, error=None):
        if self.s2_due is None:
            raise CreditContractError("S2 is due only after an actual multiple of twenty optimizer updates")
        self.holdout.verify_unchanged()
        expected = {row[0]: row for row in self.holdout.episodes}
        seen = set()
        for row in episodes:
            identity = row.get("eval_episode_id")
            if identity not in expected or identity in seen:
                raise CreditContractError("S2 episode is outside the fixed holdout or repeated")
            reference = expected[identity]
            if (tuple(row.get(key) for key in ("eval_episode_id", "pair_id", "scene", "world_id", "seed")) != reference
                    or type(row.get("success")) is not bool):
                raise CreditContractError("S2 requires actual per-episode success and immutable holdout identity")
            seen.add(identity)
        if error is None and seen != set(expected):
            raise CreditContractError("Missing S2 episodes cannot become an observed success-rate point")
        self._append({"kind": "s2_evaluation", **self.s2_due, "episodes": episodes,
                      "complete": error is None, "error": error, "missing_eval_episode_ids": sorted(set(expected) - seen),
                      "success_rate": None, "auc": None})
        self.s2_due = None

    def close(self, *, error=None):
        if self.closed:
            raise CreditContractError("Recorder already closed")
        self._append({"kind": "session_end", "actual_completed_optimizer_updates": self.completed_updates,
                      "update_attempts": self.attempts, "unclosed_update_attempt": self.active,
                      "unrecorded_s2_due": self.s2_due, "error": error})
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.stream.close()
        self.closed = True
        index = {"identity": self.identity, "actual_completed_optimizer_updates": self.completed_updates,
                 "complete_recording": self.active is None and self.s2_due is None and error is None,
                 "files": [{"path": str(path), "bytes": path.stat().st_size, "sha256": file_digest(path)} for path in self.chunks]}
        with (self.directory / "index.json").open("x") as stream:
            json.dump(index, stream, indent=2, allow_nan=False)
        return index

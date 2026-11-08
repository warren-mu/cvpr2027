"""Audit and merge approved-policy T01 candidates; retain old evidence as ledger only.

Use run_logged after commit. Parent collection JSONs expand ONLY their recorded
scene runs, never a directory glob. --retry-of NEW_RESULT=OLD_RESULT authorizes
overlap between explicitly supplied attempts of the same logical partition;
all outcomes remain in the ledger and conflicting duplicate outcomes are flagged.
Writes only isolated named pilot/pending chunks, never data/world_pairs/pilot.jsonl.
All arithmetic and file output must execute through run_logged after commit.
"""
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import re
import shutil

from src.envs.pair_approved_policy import POLICY, evidence_passes, position_distance

from src.envs.thor_common import ROOT, write_result
from src.tools.t01_partition_approved import (C_MANIFEST, DEFAULT_MANIFEST, audited_result, candidate_identity, candidate_outcome,
                                             canonical_hash, load_manifest, require_same_scene_ownership, sha_file)
from src.tools.t01_partition import DEFAULT_MANIFEST as LEGACY_MANIFEST, load_manifest as load_legacy_manifest

KINDS = ("direct_flip", "direct_stay", "look_flip", "look_stay")
REQUIRED = ("restore_deterministic", "valid_physical", "unique_targets", "witness_replay", "evidence_available", "targets_separable")


def measurement_policy(config, measured):
    if config.get("collect_only") is True:
        return "legacy_collect_only"
    if config.get("collect_only") is False and config.get("acceptance_policy") == POLICY:
        return POLICY["id"]
    if (config.get("collect_only") is False and not measured.get("output_files")
            and measured.get("attempts") in (None, 0) and measured.get("status") == "failed_before_generation"):
        return "unavailable_before_generation"
    raise ValueError("Source has no recognized exact measurement policy")


def finite_nonnegative(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def require_source_manifest(record, collect_only, approved_ref, legacy_ref, *, partition=None):
    if type(collect_only) is not bool:
        raise ValueError("Source must identify its collection mode")
    expected = [legacy_ref] if collect_only else (approved_ref if isinstance(approved_ref, list) else [approved_ref])
    matches = [item for item in expected if isinstance(record, dict)
               and all(record.get(key) == item[key] for key in ("sha256", "manifest_id"))]
    if len(matches) != 1:
        raise ValueError("Source manifest does not match one exact registered v1/v2/v3 measurement policy")
    matched = matches[0]
    if partition is not None and not collect_only:
        permitted = {2: ("a", "b"), 3: ("legacy",)}.get(matched.get("schema_version"), ())
        if partition not in permitted:
            raise ValueError("Approved source partition is not authorized by its exact manifest version")
    return matched


def validate_leaf_scope(config, scenes, partition, manifest, approved_refs, legacy_ref):
    """C has one-scene leaves; the historical six-scene run is ledger-only."""
    if partition not in manifest["partitions"] or any(
            scene not in manifest["partitions"][partition]["scenes"] for scene in scenes):
        raise ValueError("Leaf scene is outside its fixed logical owner")
    if "shard" in config:
        shard = config["shard"]
        if not isinstance(shard, dict):
            raise ValueError("Invalid shard metadata")
        require_source_manifest(shard.get("manifest"), config.get("collect_only"),
                                approved_refs, legacy_ref, partition=partition)
        if (len(scenes) != 1 or config.get("pilot_scenes") != scenes or shard.get("scene") != scenes[0]
                or shard.get("partition") != partition):
            raise ValueError("Shard metadata does not match its parent/manifest scene")
    elif (partition != "legacy" or scenes != manifest["partitions"]["legacy"]["scenes"]
          or config.get("pilot_scenes") != scenes or config.get("collect_only") is not True):
        raise ValueError("Only the historical collect-only source may be a monolithic six-scene legacy run")


def approved_reasons(row, policy):
    """Recheck recorded scientific boundaries; absent measurements never become passes."""
    if policy != POLICY["id"]:
        return ["legacy_or_unavailable_measurement_policy"]
    reasons = []
    if row.get("acceptance_policy") != POLICY:
        reasons.append("row_policy_mismatch")
    if row.get("status") != "passed":
        reasons.append("source_did_not_pass")
    if row.get("discard_reasons") != [] or row.get("seed") != 0:
        reasons.append("nonempty_discard_reasons_or_wrong_seed")
    validity = row.get("validity", {})
    if (not isinstance(validity, dict) or not all(validity.get(key) is True for key in REQUIRED)
            or any(value is not True for value in validity.values())
            or any(row.get(key) is not True for key in ("restore_deterministic", "valid_physical", "targets_separable"))):
        reasons.append("missing_or_failed_validity")
    kind, prefix = row.get("kind"), row.get("prefix_comparison", {})
    look = isinstance(kind, str) and kind.startswith("look_")
    if kind not in KINDS or row.get("history") != "empty before single initial frame":
        reasons.append("unsupported_kind_or_history")
    if look:
        if (row.get("same_input_at_t") is not True or row.get("same_input_at_t_strict") is not True
                or any(prefix.get(key) is not True for key in ("same_input_at_t_strict", "rgb_exact", "marked_rgb_exact", "text_exact"))
                or any(prefix.get(key) != 0 or type(prefix.get(key)) is not int
                       for key in ("rgb_max_abs_diff", "marked_max_abs_diff", "prefix_max_abs_diff", "different_pixels"))
                or row.get("prefix_max_abs_diff") != 0):
            reasons.append("look_prefix_not_exact")
        action = row.get("common_observation_action")
        if (not isinstance(action, str) or not action or action.startswith("select")
                or row.get("immediate_action_labels") != [action, action]
                or row.get("immediate_target_labels") is not None):
            reasons.append("look_immediate_labels_not_shared_observation")
    elif row.get("same_input_at_t_strict") is not False or prefix.get("same_input_at_t_strict") is not False:
        reasons.append("direct_prefix_not_distinguishable")
    names = [row.get("target_name_w0"), row.get("target_name_w1")]
    if (not all(isinstance(value, str) and value for value in names)
            or (kind in KINDS and (names[0] != names[1]) != kind.endswith("flip"))):
        reasons.append("target_identity_or_class_mismatch")
    evidence = row.get("evidence_comparison", {})
    try:
        height, width = evidence["evidence_resolution"]
        count = evidence["relevant_difference_pixels"]
        if (evidence.get("acceptance_policy") != POLICY["id"]
                or type(evidence.get("evidence_channel_difference_exclusive")) is not int
                or evidence["evidence_channel_difference_exclusive"] != 10
                or evidence.get("evidence_passed") is not True
                or row.get("evidence_pixels_after_look") != count
                or type(evidence.get("relevant_mask_pixels")) is not int
                or not count <= evidence["relevant_mask_pixels"] <= height * width
                or not evidence_passes(count, height, width)):
            raise ValueError("Not approved evidence")
    except (KeyError, TypeError, ValueError):
        reasons.append("missing_or_insufficient_strict_evidence")
    physical = row.get("physical_checks", [])
    try:
        if len(physical) != 2:
            raise ValueError("Need both settled worlds")
        for check in physical:
            collisions = check["overlapping_colliders"]
            settling = check["settling"]
            last = settling.get("last_observation") or (settling.get("observations") or [None])[-1]
            if (check.get("valid_physical") is not True or check.get("native_collision_clear") is not True
                    or check.get("errors") != [] or not isinstance(collisions, dict) or not collisions
                    or any(not isinstance(value, list) or value for value in collisions.values())
                    or settling.get("settled") is not True or not isinstance(last, dict)
                    or last.get("native_isSceneAtRest") is not True or last.get("stable") is not True
                    or last.get("inventory_exact") is not True
                    or type(last.get("consecutive_stable")) is not int
                    or last["consecutive_stable"] < 5):
                raise ValueError("Incomplete native collision/settling evidence")
        edited = physical[1]
        if (edited.get("acceptance_policy") != POLICY or edited.get("inventory_exact") is not True
                or edited.get("missing") != [] or edited.get("added") != []
                or edited.get("position_bounds_passed") is not True
                or edited.get("legal_receptacle_placement") is not True
                or edited.get("destination_member_after_settling") is not True
                or not isinstance(edited.get("destination_contains_after_settling"), list)
                or not edited["destination_contains_after_settling"]
                or row["edit"].get("native_proposal_verified") is not True
                or row["edit"].get("proposal_pose_changed") is not True
                or row["edit"].get("proposal_receptacle_member") is not True):
            raise ValueError("Missing native legal placement/inventory evidence")
        error, collateral = edited["edited_settling_position_error_m"], edited["maximum_other_object_displacement_m"]
        if not finite_nonnegative(error) or error > .02 or not finite_nonnegative(collateral) or collateral > .01:
            raise ValueError("Measured physical bound exceeded or unknown")
        deltas, name = edited["all_object_pose_deltas"], row["edit"]["object_name"]
        measured_error = position_distance(row["edit"]["requested_pose"]["position"], deltas[name]["after_position"])
        if not math.isclose(measured_error, error, rel_tol=1e-12, abs_tol=1e-12) or measured_error > .02:
            raise ValueError("Settling error contradicts measured positions")
        others = []
        for object_name, delta in deltas.items():
            distance = position_distance(delta["before_position"], delta["after_position"])
            if not finite_nonnegative(delta["position_delta_m"]) or not math.isclose(distance, delta["position_delta_m"], rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("Object displacement contradicts positions")
            if object_name != name:
                others.append(distance)
        if not math.isclose(max(others, default=0), collateral, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("Collateral maximum contradicts measured object positions")
        inventories = [{pose["objectName"] for pose in row[key]["poses"]} for key in ("w0", "w1", "requested_w1")]
        if not inventories[0] or not inventories[0] == inventories[1] == inventories[2] or not inventories[0] <= set(deltas):
            raise ValueError("Complete editable inventory is missing")
        if any(len(inventory) != len(row[key]["poses"]) for inventory, key in zip(inventories, ("w0", "w1", "requested_w1"))):
            raise ValueError("Duplicate requested/realized pose identity")
        before = {pose["objectName"]: pose for pose in row["w0"]["poses"]}
        requested = {pose["objectName"]: pose for pose in row["requested_w1"]["poses"]}
        if requested[name] != row["edit"]["requested_pose"] or any(requested[key] != value for key, value in before.items() if key != name):
            raise ValueError("Requested edit is not exactly one object")
    except (KeyError, TypeError, ValueError, IndexError):
        reasons.append("missing_or_failed_measured_physics")
    try:
        restores, witnesses = row["restore_checks"], row["witnesses"]
        if len(restores) != 2 or len(witnesses) != 2:
            raise ValueError("Missing two-world replay")
        for index, (restore, witness) in enumerate(zip(restores, witnesses)):
            if (restore.get("passed") is not True or restore["comparison"].get("same_input_at_t_strict") is not True
                    or witness["plan"].get("found") is not True):
                raise ValueError("Unverified restore or plan")
            replay = witness["replay"]
            actions = replay["actions"]
            terminal = actions[-1]
            if (not 1 <= len(actions) <= 12 or row[f"witness_actions_w{index}"] != actions
                    or replay.get("passed") is not True or replay.get("selection_success") is not True
                    or replay.get("prefix_replay_exact") is not True
                    or (look and replay.get("first_observation_exact") is not True)
                    or (look and actions[0].get("action") != row["common_observation_action"])
                    or terminal.get("success") is not True or not terminal.get("action", "").startswith("select ")
                    or terminal.get("target_name") != names[index] or terminal.get("resolved_target_name") != names[index]
                    or not finite_nonnegative(terminal.get("distance_m")) or terminal["distance_m"] > 1.5):
                raise ValueError("Incomplete successful witness")
    except (KeyError, TypeError, ValueError, IndexError):
        reasons.append("missing_or_failed_restore_witness")
    return reasons


def verify_raw_attempt(row):
    """Require the hashed full evidence from which the bounded journal was made."""
    from src.envs.pair_generator_approved import compact_record
    ref = row["raw_attempt_evidence"]
    path = Path(ref["path"])
    if not path.is_absolute() or path.stat().st_size != ref["bytes"] or sha_file(path) != ref["sha256"]:
        raise ValueError("Full attempt evidence size/hash mismatch")
    raw = json.loads(path.read_text())
    if compact_record(raw, path) != row:
        raise ValueError("Journal does not exactly match its authenticated full attempt")
    for key in ("w0", "w1", "requested_w1"):
        objects = raw[key]["objects"]
        if len({obj["name"] for obj in objects}) != len(objects):
            raise ValueError("Duplicate stable name in full scene metadata")
        editable = {obj["name"] for obj in objects if obj.get("moveable") is True or obj.get("pickupable") is True}
        if {pose["objectName"] for pose in raw[key]["poses"]} != editable:
            raise ValueError("Pose list omits/adds an editable scene object")
    queries = Path(row["query_log"])
    if not queries.is_absolute() or sha_file(queries) != row["query_log_sha256"]:
        raise ValueError("Native query evidence hash mismatch")


def approved_outcome(row):
    fields = ("edited_settling_position_error_m", "maximum_other_object_displacement_m",
              "all_object_pose_deltas", "inventory_exact", "missing", "added", "legal_receptacle_placement",
              "destination_member_after_settling", "native_collision_clear", "overlapping_colliders", "errors")
    return canonical_hash({"original_outcome": candidate_outcome(row),
                           "physical_measurements": [{key: item.get(key) for key in fields}
                                                      for item in row.get("physical_checks", [])]})


def add_candidate(groups, identity, row, occurrence):
    item = groups.setdefault(identity, {"occurrences": [], "kinds": set(), "statuses": set(), "outcomes": set(),
                                        "row": None, "policies": set(), "approved_occurrences": []})
    policy = occurrence["measurement_policy"]
    item["occurrences"].append(occurrence)
    item["policies"].add(policy)
    if policy == POLICY["id"]:
        item["approved_occurrences"].append(occurrence)
        item["kinds"].add(row["kind"])
        item["statuses"].add(row.get("status"))
        item["outcomes"].add(approved_outcome(row))
        if item["row"] is None:
            item["row"] = row


class SelectionChunks:
    """Streaming portable JSONL, each <=32MiB; unique named directory only."""
    def __init__(self, directory, prefix, limit=32 * 1024 * 1024):
        self.directory, self.prefix, self.limit = directory, prefix, limit
        directory.mkdir(parents=True, exist_ok=False)
        self.paths, self.stream, self.size = [], None, 0

    def write(self, row):
        encoded = (json.dumps(row, sort_keys=True, allow_nan=False) + "\n").encode()
        if len(encoded) > self.limit:
            raise ValueError("One selected row exceeds portable evidence size limit")
        if self.stream is None or self.size + len(encoded) > self.limit:
            if self.stream:
                self.stream.close()
            path = self.directory / f"{self.prefix}_{len(self.paths):04d}.jsonl"
            self.stream = path.open("xb")
            self.paths.append(path)
            self.size = 0
        self.stream.write(encoded)
        self.stream.flush()
        self.size += len(encoded)

    def close(self):
        if not self.paths:
            path = self.directory / f"{self.prefix}_0000.jsonl"
            with path.open("xb"):
                pass
            self.paths.append(path)
        if self.stream:
            self.stream.close()
        return [{"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha_file(path)} for path in self.paths]


class JournalChunks:
    def __init__(self, directory, limit=32 * 1024 * 1024):
        self.directory, self.limit = directory, limit
        self.stream, self.size, self.paths = None, 0, []

    def write(self, value):
        encoded = (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()
        if len(encoded) > self.limit:
            raise ValueError("One source attempt exceeds the bounded merged chunk size")
        if self.stream is None or self.size + len(encoded) > self.limit:
            if self.stream:
                self.stream.close()
            path = self.directory / f"attempts_{len(self.paths):04d}.jsonl"
            self.stream, self.size = path.open("xb"), 0
            self.paths.append(path)
        self.stream.write(encoded)
        self.stream.flush()
        self.size += len(encoded)

    def close(self):
        if self.stream:
            self.stream.close()
        return [{"path": str(path), "bytes": path.stat().st_size, "sha256": sha_file(path)} for path in self.paths]


def retry_roots(origins, declarations):
    parents = {}
    for declaration in declarations:
        new, separator, old = declaration.partition("=")
        if not separator:
            raise ValueError("Use --retry-of NEW_RESULT=OLD_RESULT")
        new, old = str((ROOT / new).resolve()), str((ROOT / old).resolve())
        if new not in origins or old not in origins or new == old or new in parents:
            raise ValueError("Retry endpoints must be distinct explicit sources, each retry declared once")
        parents[new] = old
    roots = {}
    for origin in origins:
        current, seen = origin, set()
        while current in parents:
            if current in seen:
                raise ValueError("Retry declarations contain a cycle")
            seen.add(current)
            current = parents[current]
        roots[origin] = current
    return roots


def journal_path(data):
    path = (ROOT / data["config"]["output_paths"]["all"]).resolve()
    records = [item for item in data["metrics"].get("output_files", [])
               if (ROOT / item["path"]).resolve() == path]
    if len(records) != 1 or path.suffix != ".jsonl" or sha_file(path) != records[0]["sha256"]:
        raise ValueError("Raw attempt journal is missing a unique matching logged hash")
    return path, records[0]["sha256"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--source", nargs="+", action="extend", required=True)
    parser.add_argument("--retry-of", action="append", default=[])
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--scratch", type=Path, default=None)
    args = parser.parse_args()
    if not re.fullmatch(r"pilot_pairs_[A-Za-z0-9_]+", args.name):
        parser.error("Use a fresh pilot_pairs_ merge name")
    output = ROOT / "results" / f"{args.name}.json"
    if output.exists():
        raise FileExistsError("Existing merged evidence is preserved")
    metrics = {"kind": "t01_merged_approved", "aggregation_status": "failed", "acceptance_passed": False,
               "source_runs": [], "collection_runs": [], "missing_shard_results": []}
    config = {"name": args.name, "requested_sources": args.source, "retry_of": args.retry_of,
              "collect_only": False, "thresholds_applied": True, "acceptance_policy": POLICY,
              "notes": ["Only independently rechecked exact approved-policy passed rows enter this isolated pilot.",
                        "Legacy collect-only rows remain in the ledger and can never be promoted by this merger.",
                        "Every explicit failure remains in source_runs/collection_runs and every available journal row remains in scratch chunks.",
                        "Scene overlap is rejected except for explicitly linked retries within the same declared logical owner.",
                        "Duplicate physical pair proposals count once; conflicting measurements are reported separately, never resolved by best outcome.",
                        "Unknown pair identity is retained as an unidentified attempt, not counted as a unique candidate.",
                        "Review selection is first ten approved unique candidates per class in fixed numeric-scene/attempt order; pending rows are separate.",
                        "PBS job IDs are listed for separate final accounting; no allocation usage or credit charge is inferred here."]}
    sources, cache, writer, groups = [], {}, None, {}
    pilot_writer = pending_writer = None
    errors, warnings = [], []
    code = 1
    try:
        manifest, manifest_ref = load_manifest(args.manifest)
        legacy_manifest, legacy_ref = load_legacy_manifest(LEGACY_MANIFEST)
        registered = [load_manifest(path) for path in (DEFAULT_MANIFEST, C_MANIFEST)]
        approved_refs = [reference for _, reference in registered]
        for other_manifest, _ in registered:
            require_same_scene_ownership(manifest, other_manifest)
        require_same_scene_ownership(manifest, legacy_manifest)
        require_source_manifest(manifest_ref, False, approved_refs, legacy_ref)
        config["manifest"], config["legacy_ledger_manifest"] = manifest_ref, legacy_ref
        config["approved_manifest_catalog"] = approved_refs
        paths = [str((ROOT / path).resolve()) for path in args.source]
        if len(paths) != len(set(paths)):
            raise ValueError("Duplicate explicit source path")
        roots = retry_roots(set(paths), args.retry_of)

        def load(path):
            key = str((ROOT / path).resolve())
            if key not in cache:
                data, ref = audited_result(key)
                cache[key] = (data, ref)
                sources.append(ref)
            return cache[key]

        leaves, ownership, root_partitions = [], {}, {}
        for origin in paths:
            data, ref = load(origin)
            is_collection = data["metrics"].get("kind") == "t01_parallel_collect"
            partition = data["config"].get("partition") if is_collection else data["config"].get("shard", {}).get("partition", "legacy")
            if partition not in manifest["partitions"]:
                raise ValueError("Unknown partition in explicit source")
            if roots[origin] in root_partitions and root_partitions[roots[origin]] != partition:
                raise ValueError("Retries cannot change partition")
            root_partitions[roots[origin]] = partition
            declared = data["config"].get("pilot_scenes", [])
            if is_collection:
                require_source_manifest(data["config"].get("manifest"), data["config"].get("collect_only"),
                                        approved_refs, legacy_ref, partition=partition)
                if declared != manifest["partitions"][partition]["scenes"]:
                    raise ValueError("Collection scene manifest differs from the declared partition")
                if not (data["config"].get("collect_only") is True or
                        (data["config"].get("collect_only") is False and data["config"].get("acceptance_policy") == POLICY["id"])):
                    raise ValueError("Unknown parallel collection measurement policy")
                metrics["collection_runs"].append({"source": ref, "partition": partition,
                    "pbs_jobid": data["config"].get("pbs_jobid"), "status": data["metrics"].get("status"),
                    "acceptance_policy": data["config"].get("acceptance_policy"), "collect_only": data["config"].get("collect_only"),
                    "gate": data["metrics"].get("gate"), "unstarted_scenes": data["metrics"].get("unstarted_scenes"),
                    "unfinished_scenes": data["metrics"].get("unfinished_scenes"),
                    "cleanup_errors": data["metrics"].get("cleanup_errors")})
                parent_refs = {item["file"]: item for item in data.get("sources", [])}
                children = data["metrics"].get("scene_runs", [])
                if len({row["scene"] for row in children}) != len(children):
                    raise ValueError("One collection attempted a scene more than once")
                for child in children:
                    child_path = child["result_path"]
                    if child.get("provenance_error") or child_path not in parent_refs:
                        metrics["missing_shard_results"].append({"collection": ref["file"], **child})
                        continue
                    leaf, leaf_ref = load(child_path)
                    if leaf_ref["sha256"] != parent_refs[child_path]["sha256"]:
                        raise ValueError("Recorded child result changed since collection finalized")
                    if (leaf["config"].get("collect_only") is not data["config"].get("collect_only")
                            or leaf["config"].get("shard", {}).get("pbs_jobid") != data["config"].get("pbs_jobid")):
                        raise ValueError("Child policy mode/PBS ID differs from its recorded parent")
                    leaves.append((leaf, leaf_ref, origin, partition, [child["scene"]]))
            else:
                leaves.append((data, ref, origin, partition, declared))
            for scene in declared:
                if scene not in manifest["partitions"][partition]["scenes"]:
                    raise ValueError(f"Scene {scene} assigned to the wrong logical owner")
                if scene in ownership and ownership[scene] != roots[origin]:
                    raise ValueError(f"Scene {scene} overlaps across sources without explicit --retry-of")
                ownership[scene] = roots[origin]

        scratch_value = args.scratch or os.environ.get("CVPR_SCRATCH")
        if not scratch_value:
            raise ValueError("Supply --scratch or CVPR_SCRATCH for full merged journals")
        scratch = Path(scratch_value).resolve() / "t01_merged_approved" / args.name
        scratch.mkdir(parents=True, exist_ok=False)
        config["merged_scratch_directory"] = str(scratch)
        writer = JournalChunks(scratch)
        portable = ROOT / "data/world_pairs" / args.name
        if portable.exists():
            raise FileExistsError("Named pilot/pending evidence already exists")
        pilot_writer = SelectionChunks(portable / "pilot", "pilot")
        pending_writer = SelectionChunks(portable / "pending_review", "pending_review")
        config["output_directories"] = {"pilot": str((portable / "pilot").relative_to(ROOT)),
                                        "pending_review": str((portable / "pending_review").relative_to(ROOT))}
        all_attempts, unidentified, leaf_paths = 0, 0, set()
        raw_counts = {kind: Counter() for kind in KINDS}
        raw_reasons = {kind: Counter() for kind in KINDS}
        observed_seconds = []
        for data, ref, origin, partition, scenes in leaves:
            if ref["file"] in leaf_paths:
                raise ValueError("Same leaf supplied both directly and through a collection")
            leaf_paths.add(ref["file"])
            cfg, measured = data["config"], data["metrics"]
            if cfg.get("seed") != manifest["seed"]:
                raise ValueError("Only the predeclared seed-zero evidence is supported")
            policy = measurement_policy(cfg, measured)
            validate_leaf_scope(cfg, scenes, partition, manifest, approved_refs, legacy_ref)
            record = {"source": ref, "origin": str(Path(origin).relative_to(ROOT)), "partition": partition,
                      "scenes": scenes, "pbs_jobid": cfg.get("runtime", {}).get("pbs_jobid") or cfg.get("shard", {}).get("pbs_jobid"),
                      "status": measured.get("status"), "recorded_attempts": measured.get("attempts"),
                      "measurement_policy": policy,
                      "counts_by_class": measured.get("counts_by_class"), "scene_failures": measured.get("scene_failures"),
                      "error": measured.get("error"), "cleanup_errors": measured.get("cleanup_errors"), "journal": None}
            metrics["source_runs"].append(record)
            if not measured.get("output_files") and measured.get("attempts") in (None, 0):
                record["journal_unavailable_reason"] = "source reports no attempts or unknown attempts and no hashed journal"
                continue
            journal, digest = journal_path(data)
            record["journal"] = {"path": str(journal), "sha256": digest}
            seen_ids, rows = set(), 0
            with journal.open() as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    attempt = row.get("attempt_id")
                    if not isinstance(attempt, str) or attempt in seen_ids or row.get("scene") not in scenes or row.get("kind") not in KINDS:
                        raise ValueError("Duplicate/invalid attempt identity, scene or kind in journal")
                    seen_ids.add(attempt)
                    identity = candidate_identity(row)
                    occurrence = {"source_file": ref["file"], "source_sha256": ref["sha256"],
                                  "measurement_policy": policy,
                                  "attempt_id": attempt, "candidate_identity": identity,
                                  "global_attempt_id": canonical_hash([ref["sha256"], attempt])}
                    writer.write({**occurrence, "original": row})
                    rows += 1
                    all_attempts += 1
                    raw_counts[row["kind"]][row.get("status") or "unknown"] += 1
                    raw_reasons[row["kind"]].update(row.get("discard_reasons", []))
                    elapsed = row.get("generation_seconds")
                    if type(elapsed) in (int, float) and math.isfinite(elapsed) and elapsed >= 0:
                        observed_seconds.append(elapsed)
                    if identity is None:
                        unidentified += 1
                        continue
                    add_candidate(groups, identity, row, occurrence)
            if measured.get("attempts") is not None and rows != measured["attempts"]:
                raise ValueError("Journal length disagrees with the logged attempt count")
            record["journal_rows_verified"] = rows

        counts = {kind: Counter() for kind in KINDS}
        conflicts, duplicates, reviews = [], [], []
        accepted, pending, rejected_by_merger, cross_policy = [], [], [], []
        look_observed, look_strict = 0, 0
        low_diff = {kind: 0 for kind in KINDS if kind.startswith("look_")}
        for identity, item in groups.items():
            if len(item["occurrences"]) > 1:
                duplicates.append({"candidate_identity": identity, "occurrences": item["occurrences"]})
            if len(item["policies"]) > 1:
                cross_policy.append({"candidate_identity": identity, "measurement_policies": sorted(item["policies"]),
                                     "occurrences": item["occurrences"]})
            if not item["approved_occurrences"]:
                continue
            if len(item["kinds"]) != 1 or len(item["statuses"]) != 1 or len(item["outcomes"]) != 1:
                conflicts.append({"candidate_identity": identity, "kinds": sorted(item["kinds"]),
                                  "statuses": sorted(item["statuses"], key=str), "outcome_hashes": sorted(item["outcomes"]),
                                  "occurrences": item["occurrences"]})
                continue
            kind, status = next(iter(item["kinds"])), next(iter(item["statuses"]))
            counts[kind][status or "unknown"] += 1
            row = item["row"]
            if kind.startswith("look_"):
                if type(row.get("same_input_at_t_strict")) is bool:
                    look_observed += 1
                    look_strict += int(row["same_input_at_t_strict"])
                difference = row.get("prefix_max_abs_diff")
                if type(difference) in (int, float) and 0 < difference <= 3:
                    low_diff[kind] += 1
            if status == "passed":
                reasons = approved_reasons(row, POLICY["id"])
                if not reasons:
                    try:
                        verify_raw_attempt(row)
                    except (OSError, ValueError, KeyError, TypeError) as error:
                        reasons.append("unverified_full_attempt: " + str(error))
                if reasons:
                    rejected_by_merger.append({"candidate_identity": identity, "reasons": reasons,
                                               "occurrences": item["approved_occurrences"]})
                else:
                    accepted.append((identity, row, item["approved_occurrences"]))
                    reviews.append((identity, row))
            elif status == "pending_review":
                # Preserve pending evidence exactly; it is never a passed pair.
                pending.append((identity, row, item["approved_occurrences"]))
        order = lambda item: (KINDS.index(item[1]["kind"]), int(item[1]["scene"].removeprefix("FloorPlan")), item[1]["attempt_id"], item[0])
        for rows_to_write, target in ((accepted, pilot_writer), (pending, pending_writer)):
            for identity, row, occurrences in sorted(rows_to_write, key=order):
                target.write({**row, "merge_provenance": {"candidate_identity": identity, "occurrences": occurrences,
                                                        "acceptance_policy": POLICY["id"]}})
        review_dir = ROOT / "logs/review/pilot" / args.name
        selected = []
        for kind in KINDS:
            candidates = sorted(((identity, row) for identity, row in reviews if row["kind"] == kind),
                                key=lambda pair: (int(pair[1]["scene"].removeprefix("FloorPlan")), pair[1]["attempt_id"], pair[0]))[:10]
            for index, (identity, row) in enumerate(candidates):
                selection = {"candidate_identity": identity, "kind": kind, "source_image": row.get("review_image"),
                             "sha256": row.get("review_image_sha256"), "copied_to": None}
                if row.get("review_image") and row.get("review_image_sha256"):
                    image = Path(row["review_image"])
                    if image.suffix.lower() not in (".jpg", ".jpeg") or sha_file(image) != row["review_image_sha256"]:
                        raise ValueError("Selected review image hash/type mismatch")
                    review_dir.mkdir(parents=True, exist_ok=True)
                    destination = review_dir / f"{kind}_{index:02d}_{identity[:12]}.jpg"
                    with image.open("rb") as incoming, destination.open("xb") as outgoing:
                        shutil.copyfileobj(incoming, outgoing)
                    selection["copied_to"] = str(destination.relative_to(ROOT))
                else:
                    selection["unavailable_reason"] = "source has no authenticated review JPEG; no later candidate substituted"
                selected.append(selection)
        metrics.update(all_available_attempt_rows=all_attempts, unidentified_attempt_rows=unidentified,
                       raw_attempt_counts_by_class={k: dict(v) for k, v in raw_counts.items()},
                       raw_discard_reason_counts={k: dict(v) for k, v in raw_reasons.items()},
                       mean_observed_attempt_generation_seconds=(sum(observed_seconds) / len(observed_seconds) if observed_seconds else None),
                       generation_time_observation_count=len(observed_seconds),
                       generation_time_denominator="all logged attempts with a finite nonnegative duration, including retries and rejections; not successful-pair cost",
                       unique_unconflicted_look_prefix_comparisons=look_observed,
                       unique_unconflicted_look_strict_prefix_count=look_strict,
                       unique_unconflicted_look_strict_prefix_fraction=(look_strict / look_observed if look_observed else None),
                       unique_unconflicted_low_diff_look_counts=low_diff,
                       low_diff_count_scope="measured 0 < prefix_max_abs_diff <= 3 only; not acceptance or a waiver of other checks",
                       unique_identified_pair_count=len(groups), unique_counts_by_class={k: dict(v) for k, v in counts.items()},
                       unique_measurement_counts_policy=POLICY["id"],
                       raw_attempt_counts_scope="all explicit source policies including legacy; no legacy row is accepted",
                       duplicate_groups=duplicates, conflicting_duplicate_groups=conflicts, review_selection=selected,
                       cross_policy_duplicate_groups=cross_policy, rejected_approved_pass_rows=rejected_by_merger,
                       approved_unique_passed_by_class={kind: sum(row["kind"] == kind for _, row, _ in accepted) for kind in KINDS},
                       approved_unique_pending_by_class={kind: sum(row["kind"] == kind for _, row, _ in pending) for kind in KINDS},
                       declared_scene_owners={scene: str(Path(origin).relative_to(ROOT)) for scene, origin in sorted(ownership.items())},
                       scenes_not_supplied=sorted({scene for group in manifest["partitions"].values() for scene in group["scenes"]} - set(ownership)),
                       pbs_job_ids=sorted({row["pbs_jobid"] for row in metrics["source_runs"] + metrics["collection_runs"] if row.get("pbs_jobid")}))
        metrics["aggregation_status"] = "completed" if not conflicts and not rejected_by_merger and not metrics["missing_shard_results"] else "incomplete_evidence"
        accepted_counts = metrics["approved_unique_passed_by_class"]
        ratio = accepted_counts["look_flip"] / accepted_counts["look_stay"] if accepted_counts["look_stay"] else None
        metrics["look_passed_flip_stay_ratio"] = ratio
        metrics["scientific_count_criteria_met"] = all(value >= 20 for value in accepted_counts.values()) and ratio is not None and .8 <= ratio <= 1.25
        metrics["review_images_complete"] = (len(selected) == 40 and all(row["copied_to"] is not None for row in selected))
        metrics["acceptance_passed"] = (metrics["aggregation_status"] == "completed" and metrics["scientific_count_criteria_met"]
                                         and metrics["review_images_complete"])
        metrics["human_review_status"] = "pending; automated count/check acceptance is not the required human image review"
        metrics["all_sixty_scenes_declared"] = not metrics["scenes_not_supplied"]
        metrics["all_supplied_scene_runs_succeeded"] = bool(metrics["source_runs"]) and all(
            row["source"]["exit_code"] == 0 for row in metrics["source_runs"] + metrics["collection_runs"])
        code = 0 if metrics["aggregation_status"] == "completed" else 1
    except (OSError, ValueError, KeyError, TypeError) as error:
        errors.append(f"{type(error).__name__}: {error}")
    finally:
        metrics["merged_journal_chunks"] = writer.close() if writer else []
        metrics["pilot_files"] = pilot_writer.close() if pilot_writer else []
        metrics["pending_review_files"] = pending_writer.close() if pending_writer else []
        metrics["errors"], metrics["warnings"] = errors, warnings
        write_result(args.name, metrics, config)
        output.write_text(json.dumps({"metrics": metrics, "config": config, "sources": sources}, indent=2, allow_nan=False) + "\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())

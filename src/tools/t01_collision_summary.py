"""Archive and summarize an explicit completed collision probe; no simulator.

Invoke through run_logged after commit. Program success authenticates the saved
observations; it never approves physical validity or changes a T01 predicate.
"""
import argparse
from collections import Counter
import copy
import hashlib
import io
import json
from pathlib import Path
import re

from src.tools.phase_watch import load_logged, load_module
from src.tools.t01_collision_probe import (BOOKKEEPING, SCENES, compact_case, compare_state,
                                           state_projection, strict_equal, teleport_arguments)

ROOT = Path(__file__).resolve().parents[2]
FILE_LIMIT = 32 * 1024 * 1024
TOTAL_LIMIT = 256 * 1024 * 1024
LIMITATIONS = [
    "Summary success is evidence verification, not completion of the source diagnostic or T01 acceptance.",
    "physical_validity remains null: neither collider lists nor identity Teleport success prove general nonpenetration.",
    "Overlap can include contact; source MeshCollider/disabled-collider coverage and penetration depth remain unknown.",
    "Every non-bookkeeping metadata field is compared exactly; currentTime is preserved as a separate reported difference.",
    "Euler metadata differences are reported without a new tolerance or a claim about quaternion/physical equivalence.",
    "RGB comparisons concern returned saved event images, not inferred fresh-render counts.",
    "Cases after the first stop and objects in a scene that did not settle were not measured; they are not negative collision results.",
    "Matching scene and random seed do not establish identical initial poses or motion state across runs; no repeated-state or causal claim is inferred.",
    "Portable evidence contains every file authenticated by each final worker index plus that index and its incremental log; Unity logs/caches are outside this archive scope.",
]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_new(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)


class EvidenceArchive:
    """Bounded, non-overwriting copies retaining original bytes and references."""
    def __init__(self, source_root, destination, repo, file_limit=FILE_LIMIT, total_limit=TOTAL_LIMIT):
        self.source_root, self.destination, self.repo = (Path(p).resolve() for p in (source_root, destination, repo))
        self.destination.relative_to(self.repo / "logs/runs")
        self.destination.mkdir(parents=True, exist_ok=False)
        self.file_limit, self.total_limit = file_limit, total_limit
        self.entries, self.total_bytes, self.rgb = {}, 0, {}
        self.incremental = self.destination.parent / "evidence_index.jsonl"
        write_new(self.incremental, b"")

    def add(self, reference, worker_root=None):
        if not isinstance(reference, dict):
            raise ValueError("Missing raw evidence reference")
        original = Path(reference["file"])
        path = original.resolve()
        if not original.is_absolute():
            raise ValueError("Raw evidence must use the original absolute scratch path")
        relative = path.relative_to(self.source_root)
        if worker_root is not None:
            path.relative_to(Path(worker_root).resolve())
        count = reference.get("bytes")
        if (type(count) is not int or count < 0 or count > self.file_limit
                or not re.fullmatch(r"[0-9a-f]{64}", str(reference.get("sha256")))):
            raise ValueError("Raw evidence lacks a bounded byte count and SHA-256")
        existing = self.entries.get(str(path))
        if existing is not None:
            if existing["sha256"] != reference["sha256"] or existing["bytes"] != count:
                raise ValueError("Conflicting references for the same raw evidence")
            return existing
        if self.total_bytes + count > self.total_limit:
            raise ValueError("Portable evidence total-size cap reached; prior copies and scratch originals retained")
        if path.stat().st_size != count:
            raise ValueError("Raw evidence byte count mismatch: " + str(path))
        raw = path.read_bytes()
        if len(raw) != count or sha(raw) != reference["sha256"]:
            raise ValueError("Raw evidence SHA/size mismatch: " + str(path))
        target = self.destination / relative
        write_new(target, raw)
        if sha(target.read_bytes()) != reference["sha256"]:
            raise ValueError("Portable evidence copy hash mismatch; all files retained")
        entry = {"original_file": str(original), "sha256": reference["sha256"], "bytes": count,
                 "portable_file": str(target.relative_to(self.repo)), "portable_sha256": reference["sha256"]}
        self.entries[str(path)] = entry
        self.total_bytes += count
        with self.incremental.open("ab") as stream:
            stream.write((json.dumps(entry, sort_keys=True) + "\n").encode())
        return entry

    def data(self, reference, worker_root=None):
        entry = self.add(reference, worker_root)
        return json.loads((self.repo / entry["portable_file"]).read_bytes()), entry

    def image(self, observation, worker_root):
        if observation is None:
            return None
        if not isinstance(observation, dict):
            raise ValueError("Unknown RGB observation schema")
        reference = observation.get("image")
        if reference is None:
            if observation.get("valid") is True:
                raise ValueError("Valid RGB claim has no saved image")
            return None
        entry = self.add(reference, worker_root)
        if entry["sha256"] not in self.rgb:
            from PIL import Image
            raw = (self.repo / entry["portable_file"]).read_bytes()
            with Image.open(io.BytesIO(raw)) as image:
                if image.format != "PNG" or image.mode != "RGB" or image.size != (224, 224):
                    raise ValueError("Expected a saved 224px RGB PNG")
                image.load()
                pixels = image.tobytes()
            nonzero = sum(value != 0 for value in pixels)
            self.rgb[entry["sha256"]] = {"shape": [224, 224, 3], "dtype": "uint8",
                "rgb_sha256": sha(pixels), "nonzero_values": nonzero, "min": min(pixels), "max": max(pixels),
                "black": nonzero == 0, "valid": nonzero > 0}
        measured = self.rgb[entry["sha256"]]
        if any(not strict_equal(observation.get(key), value) for key, value in measured.items()):
            raise ValueError("Saved RGB bytes disagree with the source observation")
        return {**copy.deepcopy(observation), "portable_image": entry}

    def finish(self):
        data = {"files": list(self.entries.values()), "copied_bytes": self.total_bytes,
                "file_limit_bytes": self.file_limit, "total_limit_bytes": self.total_limit,
                "source_root": str(self.source_root), "originals_retained": True}
        path = self.destination.parent / "evidence_index.json"
        body = (json.dumps(data, indent=2, allow_nan=False) + "\n").encode()
        write_new(path, body)
        return {"file": str(path.relative_to(self.repo)), "sha256": sha(body), "bytes": len(body),
                "copied_bytes": self.total_bytes, "copied_files": len(self.entries)}


def value_changes(before, after, path=""):
    """Exact typed JSON changes; no threshold, angle wrapping, or unknown-field drop."""
    if isinstance(before, dict) and isinstance(after, dict):
        changes = []
        for key in sorted(set(before) | set(after)):
            child = path + "/" + key.replace("~", "~0").replace("/", "~1")
            if key not in before or key not in after:
                changes.append({"path": child, "kind": "added" if key not in before else "removed",
                    "before_present": key in before, "after_present": key in after,
                    "before": before.get(key), "after": after.get(key)})
            else:
                changes.extend(value_changes(before[key], after[key], child))
        return changes
    if isinstance(before, list) and isinstance(after, list) and len(before) == len(after):
        return [row for index, (left, right) in enumerate(zip(before, after))
                for row in value_changes(left, right, path + "/" + str(index))]
    if strict_equal(before, after):
        return []
    return [{"path": path, "kind": "type_changed" if type(before) is not type(after) else "value_changed",
             "before_present": True, "after_present": True, "before": before, "after": after}]


def metadata_changes(before, after):
    try:
        left, right = state_projection(before), state_projection(after)
    except (ValueError, KeyError, TypeError) as error:
        return {"available": False, "reason": str(error), "objects": None, "other_fields": None,
                "clock": None, "full_state_exact": None}
    old, new = ({obj["name"]: obj for obj in side.pop("objects")} for side in (left, right))
    objects = []
    for name in sorted(set(old) | set(new)):
        if name not in old or name not in new:
            objects.append({"name": name, "before_present": name in old, "after_present": name in new,
                            "before": old.get(name), "after": new.get(name), "changes": None})
        else:
            changes = value_changes(old[name], new[name])
            if changes:
                objects.append({"name": name, "before_object_id": old[name]["objectId"],
                    "after_object_id": new[name]["objectId"], "changes": changes,
                    "position_changed": not strict_equal(old[name].get("position"), new[name].get("position")),
                    "rotation_changed": not strict_equal(old[name].get("rotation"), new[name].get("rotation"))})
    other = value_changes(left, right)
    return {"available": True, "reason": None, "objects": objects,
            "other_fields": [row for row in other if row["path"] != "/currentTime"],
            "clock": [row for row in other if row["path"] == "/currentTime"],
            "full_state_exact": not objects and not other}


def describe_action(action, reference, archive, worker_root):
    if not isinstance(action, dict) or type(action.get("index")) is not int or not isinstance(action.get("action"), str):
        raise ValueError("Unknown saved action record")
    before, after = action.get("before_metadata"), action.get("after_metadata")
    comparison, changes = None, None
    if after is not None:
        comparison = compare_state(before, after)
        if not strict_equal(comparison, action.get("state_comparison")):
            raise ValueError("Saved action metadata comparison disagrees with its full raw states")
        changes = metadata_changes(before, after)
    images = {key: archive.image(action.get(key), worker_root) for key in ("before_rgb", "after_rgb")}
    exact = (images["before_rgb"]["rgb_sha256"] == images["after_rgb"]["rgb_sha256"]
             if all(images.values()) else None)
    if exact is not None and action.get("rgb_bytes_exact") is not exact:
        raise ValueError("Saved RGB equality disagrees with authenticated decoded images")
    return {"action": action["action"], "action_index": action["index"], "arguments": action.get("arguments"),
            "lastActionSuccess": action.get("lastActionSuccess"), "errorMessage": action.get("errorMessage"),
            "actionReturn": action.get("actionReturn"), "exception": action.get("exception"),
            "reported_comparison": comparison, "metadata_changes": changes,
            "rgb_bytes_exact": exact, "source_rgb_bytes_exact": action.get("rgb_bytes_exact"),
            "rgb_unavailable_reason": "Missing/invalid saved before or after RGB" if exact is None else None,
            "rgb": images, "action_record": reference, "physical_validity": None}


def describe_case(compact, archive, worker_root):
    raw, portable = archive.data(compact["raw_case"], worker_root)
    expected = {**compact_case(raw), "raw_case": compact["raw_case"]}
    if not strict_equal(expected, compact) or raw.get("physical_validity") is not None:
        raise ValueError("Raw object case disagrees with the logged compact case")
    result = {key: copy.deepcopy(raw.get(key)) for key in
              ("object_name", "object_id", "complete", "safe_to_continue_diagnostic", "stopped_reason")}
    result.update(raw_case=portable, physical_validity=None)
    for label, action_name in (("old_query", "CollidersObjectCollidingWith"), ("identity_teleport", "TeleportObject")):
        source = raw.get(label)
        result[label] = None
        if source is None:
            continue
        action, action_ref = archive.data(source["action_record"], worker_root)
        objects = (action.get("before_metadata") or {}).get("objects", [])
        targets = [row for row in objects if row.get("name") == raw["object_name"]]
        if len(targets) != 1 or targets[0].get("objectId") != raw["object_id"]:
            raise ValueError("Case/action target identity differs")
        arguments = ({"objectId": raw["object_id"]} if label == "old_query" else teleport_arguments(targets[0]))
        if action.get("action") != action_name or not strict_equal(action.get("arguments"), arguments):
            raise ValueError("Case/action request differs from the original native comparison")
        for key in ("lastActionSuccess", "errorMessage", "actionReturn", "state_comparison", "rgb_bytes_exact"):
            if not strict_equal(source.get(key), action.get(key)):
                raise ValueError("Case fields differ from the referenced raw action: " + key)
        after = action.get("after_metadata")
        if not isinstance(after, dict) or any(not strict_equal(action.get(key), after.get(key)) for key in
                                             ("lastActionSuccess", "errorMessage", "actionReturn")):
            raise ValueError("Action response differs from its raw after metadata")
        rejected = (label == "identity_teleport" and after.get("lastActionSuccess") is False
                    and isinstance(after.get("errorMessage"), str)
                    and after["errorMessage"].startswith(raw["object_id"] + " is colliding with ")
                    and after["errorMessage"].endswith(" after teleport."))
        known = (after.get("lastAction") == action_name and type(after.get("lastActionSuccess")) is bool
                 and (after["lastActionSuccess"] or rejected))
        schema = (isinstance(after.get("actionReturn"), list) and all(isinstance(x, str) for x in after["actionReturn"])) if label == "old_query" else None
        if (source.get("native_collision_rejected") is not rejected or source.get("api_response_known") is not known
                or source.get("known_return_schema") is not schema):
            raise ValueError("Case native/API classification disagrees with the raw response")
        described = describe_action(action, action_ref, archive, worker_root)
        described.update({key: source.get(key) for key in ("known_return_schema", "api_response_known", "native_collision_rejected")})
        result[label] = described
    old, new = result["old_query"], result["identity_teleport"]
    old_state = ("not_observed" if old is None else "api_or_schema_failure" if
                 old["api_response_known"] is not True or old["known_return_schema"] is not True else
                 "nonempty" if old["actionReturn"] else "empty")
    new_state = ("not_observed" if new is None else "api_failure_unknown" if new["api_response_known"] is not True else
                 "native_collision_rejected" if new["native_collision_rejected"] is True else
                 "native_teleport_succeeded" if new["lastActionSuccess"] is True else "unknown")
    result.update(old_query_outcome=old_state, identity_teleport_outcome=new_state,
                  outcome_pair=old_state + "__" + new_state)
    return result


def summarize_worker(worker, archive):
    worker_root = Path(worker["raw_directory"]).resolve()
    row = {key: copy.deepcopy(worker.get(key)) for key in ("scene", "logical_index", "gpu_identity", "pbs_jobid", "node",
        "status", "diagnostic_completed", "black_frame_detected", "settled", "objects_expected", "errors",
        "cleanup_errors", "worker_error", "worker_returned_result")}
    row.update(physical_validity=None, cases=[], settling=None, last_settling_action=None)
    index_ref = worker.get("evidence_index")
    if index_ref is None:
        row.update(evidence_complete=False, unavailable_reason="No final worker index; not inferring missing observations")
        if worker.get("objects"):
            raise ValueError("Object observations have no final authenticated worker index")
        return row
    index, row["source_evidence_index"] = archive.data(index_ref, worker_root)
    references = index.get("files")
    if (not isinstance(references, list) or len({ref.get("file") for ref in references}) != len(references)
            or type(index.get("raw_bytes")) is not int or sum(ref["bytes"] for ref in references) != index["raw_bytes"]):
        raise ValueError("Worker raw index is missing, duplicate, or inconsistent")
    row["indexed_raw_bytes"] = index["raw_bytes"]
    actions = []
    row["initial_state_evidence"] = {key: None for key in ("source_world", "constructor", "comparison_baseline", "worker_config")}
    authenticated = {ref["file"]: ref for ref in references}
    for reference in references:
        entry = archive.add(reference, worker_root)
        stem = Path(reference["file"]).stem
        if stem in row["initial_state_evidence"]:
            row["initial_state_evidence"][stem] = entry
        if re.fullmatch(r"action_\d{5}\.json", Path(reference["file"]).name):
            action = json.loads((archive.repo / entry["portable_file"]).read_bytes())
            actions.append((action, entry))
    if worker.get("recorded_actions") is not None and len(actions) != worker["recorded_actions"]:
        raise ValueError("Recorded action count differs from the authenticated raw index")
    if worker.get("incremental_evidence_index") is not None:
        row["source_incremental_index"] = archive.add(worker["incremental_evidence_index"], worker_root)
        incremental = archive.repo / row["source_incremental_index"]["portable_file"]
        if [json.loads(line) for line in incremental.read_text().splitlines()] != references + [index_ref]:
            raise ValueError("Incremental index differs from the final authenticated file list")
    settling_ref = worker.get("settling_evidence")
    if settling_ref is not None:
        if any(row["initial_state_evidence"][key] is None for key in ("source_world", "constructor")):
            raise ValueError("Settling observation is missing its indexed original source state/constructor evidence")
        if authenticated.get(settling_ref["file"]) != settling_ref:
            raise ValueError("Settling source is not authenticated by the worker index")
        settling, entry = archive.data(settling_ref, worker_root)
        measured = settling["settling"]
        if measured.get("settled") is not worker.get("settled"):
            raise ValueError("Worker settling claim differs from its raw trace")
        observations = measured.get("observations", [])
        row["settling"] = {"settled": measured.get("settled"), "policy": measured.get("policy"),
            "observations_recorded": len(observations), "last_observation": observations[-1] if observations else None,
            "raw_evidence": entry}
        advance = [(action, entry) for action, entry in actions if action.get("action") == "AdvancePhysicsStep"]
        if advance:
            action, entry = max(advance, key=lambda pair: pair[0]["index"])
            row["last_settling_action"] = describe_action(action, entry, archive, worker_root)
    cases = worker.get("objects", [])
    if len({case.get("object_name") for case in cases}) != len(cases):
        raise ValueError("Duplicate object cases would double-count observations")
    for compact in cases:
        if authenticated.get(compact["raw_case"]["file"]) != compact["raw_case"]:
            raise ValueError("Case source is not authenticated by the worker index")
        for label in ("old_query", "identity_teleport"):
            if compact.get(label) is not None:
                ref = compact[label]["action_record"]
                if authenticated.get(ref["file"]) != ref:
                    raise ValueError("Action source is not authenticated by the worker index")
        row["cases"].append(describe_case(compact, archive, worker_root))
    observed, expected = len(row["cases"]), row["objects_expected"]
    if expected is not None and (type(expected) is not int or expected < observed):
        raise ValueError("Expected object count is invalid or below recorded cases")
    row.update(evidence_complete=True, unavailable_reason=None, observed_object_cases=observed,
        unobserved_object_cases=expected - observed if expected is not None else None,
        unobserved_reason="Source stopped or did not reach native comparisons; no inferred outcomes",
        outcome_counts=dict(Counter(case["outcome_pair"] for case in row["cases"])))
    return row


def validate_source(data, gate, gate_reference, repo):
    meta, config, metrics = data["_meta"], data["config"], data["metrics"]
    name, job = meta.get("name"), config.get("pbs_jobid")
    allocation = config.get("allocation") or {}
    raw_root = Path(allocation["scratch"]).resolve() / "t01_collision_probe" / name / job
    source = config.get("gate_source", {})
    source_path = source.get("file", source.get("path", ""))
    command = meta.get("command", [])
    if (type(meta.get("exit_code")) is not int or metrics.get("exit_code") != meta["exit_code"]
            or metrics.get("kind") != "t01_collision_probe" or metrics.get("physical_validity") is not None
            or command[1:5] != ["-m", "src.tools.t01_collision_probe", "--name", name]
            or config.get("name") != name or config.get("scenes") != list(SCENES)
            or config.get("state_bookkeeping_exclusions") != sorted(BOOKKEEPING)
            or Path(config["raw_directory"]).resolve() != raw_root or allocation.get("pbs_jobid") != job
            or (repo / source_path).resolve() != (repo / gate_reference["path"]).resolve()
            or source.get("sha256") != gate_reference["sha256"]
            or gate.get("_meta", {}).get("exit_code") != 0 or gate.get("metrics", {}).get("passed") is not True
            or gate.get("config", {}).get("allocation") != allocation
            or gate["config"].get("pbs_jobid") != job or gate["config"].get("node") != meta.get("host")):
        raise ValueError("Probe/gate provenance, allocation, source identity, or comparison policy differs")
    workers = metrics.get("workers")
    devices = allocation.get("devices", [])
    if not isinstance(workers, list) or len(workers) != 4 or len(devices) != 4:
        raise ValueError("Expected all four assigned worker records")
    for logical, (worker, device, scene) in enumerate(zip(workers, devices, SCENES)):
        if (worker.get("scene") != scene or worker.get("logical_index") != logical or worker.get("gpu_identity") != device
                or worker.get("pbs_jobid") != job or worker.get("node") != meta.get("host")
                or Path(worker["raw_directory"]).resolve() != raw_root / scene):
            raise ValueError("Worker source identity/path differs from its allocation")
    return raw_root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--source", default="results/t01_collision_pack_01_probe.json")
    args = parser.parse_args()
    if not re.fullmatch(r"t01_[A-Za-z0-9_]+", args.name):
        parser.error("Use a unique t01_ result name")
    output = ROOT / "results" / (args.name + ".json")
    directory = ROOT / "logs/runs" / args.name
    if output.exists() or output.is_symlink() or directory.exists() or directory.is_symlink():
        raise FileExistsError("Preserve existing evidence; choose a new summary name")
    metrics = {"kind": "t01_collision_summary", "aggregation_status": "failed", "source_exit_code": None,
        "source_diagnostic_completed": None, "source_black_frame_detected": None,
        "workers": [], "physical_validity": None, "task_acceptance": None, "errors": [], "evidence_index": None}
    config = {"name": args.name, "source": args.source, "limitations": LIMITATIONS,
              "file_limit_bytes": FILE_LIMIT, "total_evidence_limit_bytes": TOTAL_LIMIT,
              "state_bookkeeping_exclusions": sorted(BOOKKEEPING)}
    sources, archive = [], None
    try:
        auditor = load_module("collision_summary_auditor", ROOT / "scripts/audit_provenance.py")
        data, reference = load_logged(ROOT, ROOT / args.source, auditor)
        reference.update(file=reference["path"], audit=auditor.audit_file(str((ROOT / args.source).resolve())),
                         metrics_sha256=data["_meta"]["metrics_sha256"], exit_code=data["_meta"]["exit_code"])
        sources.append(reference)
        gate_path = data["config"]["gate_source"].get("file", data["config"]["gate_source"].get("path"))
        gate, gate_ref = load_logged(ROOT, ROOT / gate_path, auditor)
        gate_ref.update(file=gate_ref["path"], audit=auditor.audit_file(str((ROOT / gate_path).resolve())),
                        metrics_sha256=gate["_meta"]["metrics_sha256"], exit_code=gate["_meta"]["exit_code"])
        sources.append(gate_ref)
        raw_root = validate_source(data, gate, gate_ref, ROOT)
        archive = EvidenceArchive(raw_root, directory / "evidence", ROOT)
        metrics.update(source_exit_code=data["_meta"]["exit_code"], source_diagnostic_completed=data["metrics"]["diagnostic_completed"],
            source_black_frame_detected=data["metrics"]["black_frame_detected"], source_errors=data["metrics"].get("errors"),
            source_cleanup_errors=data["metrics"].get("cleanup_errors"))
        config.update(pbs_jobid=data["config"]["pbs_jobid"], source_git_commit=data["_meta"]["git_commit"],
                      source_config=data["config"], source_meta=data["_meta"])
        for worker in data["metrics"]["workers"]:
            metrics["workers"].append(summarize_worker(worker, archive))
        counts = Counter(case["outcome_pair"] for worker in metrics["workers"] for case in worker["cases"])
        metrics.update(outcome_counts=dict(counts), evidence_complete=all(worker["evidence_complete"] for worker in metrics["workers"]),
                       aggregation_status="passed")
    except Exception as error:
        metrics["errors"].append(f"{type(error).__name__}: {error}")
    finally:
        if archive is not None:
            try:
                metrics["evidence_index"] = archive.finish()
            except Exception as error:
                metrics["aggregation_status"] = "failed"
                metrics["errors"].append(f"Archive finalization: {type(error).__name__}: {error}")
        payload = {"metrics": metrics, "config": config, "sources": sources}
        body = (json.dumps(payload, indent=2, allow_nan=False) + "\n").encode()
        if len(body) > FILE_LIMIT:
            # Raw evidence and incremental index remain available for diagnosis.
            metrics = {"kind": "t01_collision_summary", "aggregation_status": "failed", "physical_validity": None,
                       "errors": ["Summary exceeds the 32MiB small-artifact limit; raw archive retained"],
                       "evidence_index": metrics["evidence_index"]}
            body = (json.dumps({"metrics": metrics, "config": {"name": args.name}, "sources": sources}, indent=2) + "\n").encode()
        write_new(output, body)
        print("FINAL_METRICS_JSON: " + json.dumps(metrics, sort_keys=True, allow_nan=False), flush=True)
    return 0 if metrics["aggregation_status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

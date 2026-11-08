"""Logged, read-only inspection of explicit ended shards; never changes acceptance."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re
import shutil
from src.tools.t01_partition import ROOT, audited_result, sha_file
from src.tools.merge_pair_v2 import option, producer_arguments, reference_path


def archive_reference(ref, directory):
    source = reference_path(ref)
    if source.suffix != ".json" or source.stat().st_size > 49_000_000:
        raise ValueError("Selected JSON exceeds the declared small-evidence archive limit")
    target = directory / (ref["sha256"] + ".json")
    if not target.exists():
        with source.open("rb") as incoming, target.open("xb") as outgoing:
            shutil.copyfileobj(incoming, outgoing)
    if sha_file(target) != ref["sha256"]:
        raise ValueError("Archived diagnostic evidence SHA mismatch")
    return {"file": str(target.relative_to(ROOT)), "sha256": ref["sha256"], "source_file": ref["file"]}


def payloads(value, path=""):
    rows = []
    if isinstance(value, dict):
        for key, child in value.items():
            location = path + "/" + key
            if key in ("coordinates", "actionReturn", "objects", "before_metadata", "after_metadata"):
                rows.append({"path": location, "canonical_json_bytes": len(json.dumps(child, sort_keys=True).encode()),
                             "elements": len(child) if isinstance(child, (list, dict)) else None})
            rows.extend(payloads(child, location))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            rows.extend(payloads(child, path + "/" + str(index)))
    return rows


def store_diagnostic(index, raw_root, cap):
    entries = [json.loads(line) for line in index.read_text().splitlines() if line.strip()]
    rows, groups = [], Counter()
    for entry in entries:
        path = reference_path(entry, within=raw_root)
        size = path.stat().st_size
        if path.parent != index.parent or size != entry.get("bytes"):
            raise ValueError("Index parent/recorded bytes mismatch: " + str(path))
        family = "frame_metadata" if path.name.endswith("_metadata.json") else re.sub(r"_[0-9].*", "", path.stem) + path.suffix
        groups[family] += size
        rows.append({**entry, "actual_bytes": size})
    largest = sorted(rows, key=lambda item: (-item["actual_bytes"], item["file"]))[:8]
    for item in largest:
        if Path(item["file"]).suffix == ".json" and item["actual_bytes"] <= 64 * 1024 * 1024:
            data = json.loads(Path(item["file"]).read_text())
            item["largest_json_subtrees"] = sorted(payloads(data), key=lambda row: -row["canonical_json_bytes"])[:12]
        else:
            item["largest_json_subtrees"] = None
    return {"index": {"file": str(index), "sha256": sha_file(index)}, "indexed_files": len(rows),
            "verified_indexed_bytes": sum(row["actual_bytes"] for row in rows), "configured_content_cap_bytes": cap,
            "index_jsonl_bytes_outside_store_counter": index.stat().st_size, "bytes_by_family": dict(groups),
            "largest_files": largest, "final_index_exists": (index.parent / "evidence_index.json").is_file()}


def first_physical_failure(config, metrics, raw):
    journal_ref = {"file": metrics["attempt_journal"], "sha256": metrics["attempt_journal_sha256"]}
    journal = reference_path(journal_ref, within=ROOT / "data/world_pairs/v2")
    if metrics["attempt_journal"] != config["attempt_journal"]:
        raise ValueError("Source journal identity mismatch")
    selected = next((json.loads(line) for line in journal.read_text().splitlines() if line.strip()
                     and "w0_physical" in (json.loads(line).get("discard_reasons") or [])), None)
    if selected is None:
        return {"status": "no_w0_physical_rejection_in_authenticated_journal", "journal": journal_ref}
    attempt_path = reference_path(selected["raw_attempt"], within=raw)
    attempt = json.loads(attempt_path.read_text())
    if any(attempt.get(key) != selected.get(key) for key in ("pair_id", "scene", "status", "discard_reasons")):
        raise ValueError("First physical rejection disagrees with its authenticated journal")
    found = []
    for line in (raw / "construction/evidence_index.jsonl").read_text().splitlines():
        ref = json.loads(line)
        if not re.fullmatch(r"construction_[0-9]+_outcome.json", Path(ref["file"]).name):
            continue
        outcome = json.loads(reference_path(ref, within=raw).read_text())
        if outcome.get("snapshot", {}).get("key") == selected["pair_id"] + "_w0":
            found.append((ref, outcome))
            break  # Only the first selected failed construction; no full attempt scan.
    if not found:
        raise ValueError("First physical rejection lacks its completed construction outcome")
    ref, outcome = found[0]
    physical = outcome["physical"]
    return {"status": "observed", "journal": journal_ref, "raw_attempt": selected["raw_attempt"],
        "pair_id": selected["pair_id"], "construction_outcome": ref, "snapshot_reference": outcome["snapshot"]["reference"],
        "policy_from_raw_attempt": attempt.get("physical_w0"),
        "backend_physical": {key: physical.get(key) for key in ("valid_physical", "edited", "edited_names",
            "collateral_position_within_1cm", "agent_unchanged", "agent_before", "agent_after",
            "native_is_scene_at_rest", "request_reference", "physical_validity_method")},
        "settling": {key: physical["settling"].get(key) for key in ("settled", "maximum_steps", "time_step_seconds", "seconds_per_step")},
        "final_settling_observation": (physical["settling"].get("observations") or [None])[-1]}


def scene_diagnostic(data):
    config, metrics, meta = data["config"], data["metrics"], data["_meta"]
    arguments = producer_arguments(meta["command"], "src.envs.pair_v2_shard")
    if (not meta.get("end") or metrics.get("kind") != "pair_v2_shard" or meta["name"] != config["name"]
            or option(arguments, "--name") != config["name"] or option(arguments, "--scene") != config["scene"]):
        raise ValueError("Require an explicit ended original v2 scene shard")
    raw = Path(config["raw_evidence_dir"]).resolve()
    if not raw.is_relative_to(Path(config["runtime"]["scratch"]).resolve()):
        raise ValueError("Raw evidence is outside the source scratch directory")
    result = {"scene": config["scene"], "status": metrics.get("status"), "exit_code": meta["exit_code"],
              "source_error": metrics.get("error"), "cleanup_errors": metrics.get("cleanup_errors"),
              "preflight": None, "first_w0_physical_failure": None, "stores": [], "diagnostic_errors": []}
    try:
        ref = metrics.get("scene_preflight_reference")
        if not ref:
            raise ValueError("Original source did not bind a scene_preflight reference")
        preflight = json.loads(reference_path(ref, within=raw).read_text())
        before = {o["name"]: o for o in preflight["diagnostics"]["source_snapshot"]["objects"]}
        after = {o["name"]: o for o in preflight["snapshot"]["objects"]}
        changes = [{"name": name, "position_delta_m": math.dist([old["position"][a] for a in "xyz"],
                    [after[name]["position"][a] for a in "xyz"]), "original_position": old["position"],
                    "final_position": after[name]["position"]} for name, old in before.items()
                   if (old.get("moveable") or old.get("pickupable")) and name in after]
        if any(not math.isfinite(row["position_delta_m"]) for row in changes):
            raise ValueError("Nonfinite original-to-final position")
        settling = preflight["diagnostics"]["settling"]
        observed = [dict(name=name, observation_index=obs["index"], **row) for obs in settling["observations"]
                    for name, row in obs["pose_measurements"]["objects"].items()]
        result["preflight"] = {"reference": ref, "reported_passed": preflight["passed"], "settled": settling["settled"],
            "final_native_rest": preflight["snapshot"].get("native_is_scene_at_rest"), "inventory_exact": set(before) == set(after),
            "missing_names": sorted(set(before) - set(after)), "added_names": sorted(set(after) - set(before)),
            "settling_observations": len(settling["observations"]), "original_to_final_positions": changes,
            "original_to_final_max_m": max((row["position_delta_m"] for row in changes), default=None),
            "final_violations_above_1cm": [row for row in changes if row["position_delta_m"] > .01],
            "recorded_transient_violations_above_1cm": [row for row in observed if row["position_delta_m"] > .01]}
    except Exception as error:
        result["diagnostic_errors"].append("preflight: " + str(error))
    try:
        result["first_w0_physical_failure"] = first_physical_failure(config, metrics, raw)
    except Exception as error:
        result["diagnostic_errors"].append("first physical rejection: " + str(error))
    indexes = sorted(raw.rglob("evidence_index.jsonl")) if raw.is_dir() else []
    if not indexes:
        result["diagnostic_errors"].append("No incremental evidence index found")
    for index in indexes:
        try:
            result["stores"].append(store_diagnostic(index, raw, config.get("max_raw_bytes")))
        except Exception as error:
            result["diagnostic_errors"].append(str(index) + ": " + str(error))
    log = ROOT / (meta.get("log_path") or f"logs/runs/{meta['name']}.log")
    lines = log.read_text(errors="replace").splitlines()
    hits = [i for i, line in enumerate(lines) if "Raw evidence path or per-worker byte limit exceeded" in line]
    result["byte_or_path_limit_error_context"] = [lines[max(0, i - 10):i + 1] for i in hits[:4]]
    result["log_reference"] = {"file": str(log.relative_to(ROOT)), "sha256": sha_file(log)}
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True); parser.add_argument("--source", nargs="+", required=True)
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9_]+", args.name): parser.error("Require a unique safe run name")
    output = ROOT / "results" / (args.name + ".json")
    archive = ROOT / "logs/runs" / args.name / "evidence"
    if output.exists() or archive.exists(): raise FileExistsError("Existing diagnostic is preserved")
    archive.mkdir(parents=True, exist_ok=False)
    metrics = {"status": "diagnostic_completed", "scenes": [], "source_errors": [], "changes_acceptance": False}
    sources, seen = [], set()
    for value in args.source:
        try:
            path = (ROOT / value).resolve()
            if path in seen: raise ValueError("Duplicate explicit source")
            seen.add(path); data, ref = audited_result(path); sources.append(ref)
            row = {"source": ref, **scene_diagnostic(data), "portable_diagnostic_evidence": []}
            metrics["scenes"].append(row)
            physical = row["first_w0_physical_failure"] or {}
            refs = [(row["preflight"] or {}).get("reference"), physical.get("raw_attempt"), physical.get("construction_outcome")]
            for selected in refs:
                if selected is not None:
                    try: row["portable_diagnostic_evidence"].append(archive_reference(selected, archive))
                    except Exception as error: row["diagnostic_errors"].append("archive: " + str(error))
        except Exception as error:
            metrics["source_errors"].append({"file": value, "error": f"{type(error).__name__}: {error}"})
    code = int(bool(metrics["source_errors"] or any(row["diagnostic_errors"] for row in metrics["scenes"])))
    if code: metrics["status"] = "diagnostic_partial"
    config = {"name": args.name, "sources_requested": args.source, "threshold_m_unchanged": .01,
              "archive_max_bytes_per_selected_json": 49_000_000,
              "scope": "Read existing bytes/metadata only; combined path/byte error does not alone prove a cap cause.",
              "payload_scope": "Largest eight saved files per worker; nested canonical subtree sizes overlap and must not be summed."}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream: json.dump({"metrics": metrics, "config": config, "sources": sources}, stream, indent=2, allow_nan=False)
    print("FINAL_METRICS_JSON: " + json.dumps(metrics, sort_keys=True, allow_nan=False), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())

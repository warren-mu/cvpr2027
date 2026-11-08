"""Report explicit, ended first-round shards; never approve pairs or submit jobs."""
import argparse
import copy
import json
from pathlib import Path
import re

from src.envs.pair_v2_clone_policy import KINDS, clone_retry_required
from src.tools.merge_pair_v2 import option, producer_arguments
from src.tools.t01_partition import ROOT, audited_result


def summarize_source(data, reference):
    """Validate source identity, then copy reported measurements without promotion."""
    config, metrics, meta = data["config"], data["metrics"], data["_meta"]
    shard = config["shard"]
    name, scene = config.get("name"), config.get("scene")
    arguments = producer_arguments(meta.get("command"), "src.envs.pair_v2_shard")
    match = re.fullmatch(r"FloorPlan([0-9]+)", str(scene))
    if (not match or int(match[1]) not in (*range(1, 31), *range(201, 231))
            or name != Path(reference["file"]).stem or meta.get("name") != name
            or option(arguments, "--name") != name or option(arguments, "--scene") != scene
            or any(shard.get(key) != config.get(key) for key in ("name", "scene", "seed", "resolution"))
            or config.get("seed") != 0 or config.get("resolution") != 255
            or config.get("enable_candidate_clone") is True or metrics.get("kind") != "pair_v2_shard"
            or type(meta.get("exit_code")) is not int or not meta.get("end")
            or metrics.get("status") in (None, "running")):
        raise ValueError("Require an ended, identity-matched first-round scene shard")
    if any("could not run git" in warning for warning in reference["audit"].get("warn", [])):
        raise ValueError("Source commit existence was not established")
    allocation, device = shard["allocation"], shard["gpu_identity"]
    logical, job, uuid = shard["logical_index"], shard["pbs_jobid"], device.get("gpu_uuid")
    devices = allocation["devices"]
    if (type(logical) is not int or not 0 <= logical < len(devices)
            or device != devices[logical] or device.get("logical_index") != logical
            or not isinstance(job, str) or not job or allocation.get("pbs_jobid") != job
            or not isinstance(uuid, str) or not uuid.startswith("GPU-")
            or shard.get("child_cuda_visible_devices") != uuid
            or shard.get("parent_cuda_visible_devices") != allocation.get("cuda_visible_devices")):
        raise ValueError("Source shard/allocation PBS or GPU identity mismatch")
    runtime = config.get("runtime")
    if runtime is not None and (runtime.get("pbs_jobid") != job or runtime.get("gpu_uuid") != uuid
                                or runtime.get("cuda_visible_devices") != uuid):
        raise ValueError("Source runtime differs from its declared PBS/GPU identity")
    counts = metrics.get("passed_by_class")
    unknown = []
    if counts is None:
        unknown = list(KINDS)
    elif not isinstance(counts, dict) or set(counts) - set(KINDS):
        raise ValueError("Unexpected reported class-count schema")
    else:
        for kind in KINDS:
            value = counts.get(kind)
            if value is None:
                unknown.append(kind)
            elif type(value) is not int or value < 0:
                raise ValueError("Reported counts must be nonnegative integers or unknown")
    attempts = metrics.get("attempts")
    if attempts is not None and (type(attempts) is not int or attempts < 0):
        raise ValueError("Reported attempts must be a nonnegative integer or unknown")
    needs_clone = None if unknown else clone_retry_required(counts)
    return {"source": copy.deepcopy(reference), "run_name": name, "scene": scene,
        "pbs_jobid": job, "gpu_uuid": uuid, "logical_index": logical,
        "runtime_identity_observed": runtime is not None,
        "reported_passed_by_class": copy.deepcopy(counts), "reported_attempts": attempts,
        "status": metrics.get("status"), "exit_code": meta["exit_code"],
        "cleanup_errors": copy.deepcopy(metrics.get("cleanup_errors")),
        "reported_discard_reason_counts": copy.deepcopy(metrics.get("discard_reason_counts")),
        "scene_excluded": copy.deepcopy(metrics.get("scene_excluded")),
        "error": copy.deepcopy(metrics.get("error")),
        "unknown_count_classes": unknown, "needs_clone_next_round": needs_clone,
        "below_threshold_classes": None if unknown else [kind for kind in KINDS if counts[kind] < 5]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--source", nargs="+", action="extend", required=True)
    args = parser.parse_args(argv)
    if re.fullmatch(r"[A-Za-z0-9_]+", args.name) is None:
        parser.error("Use a unique result name containing letters, digits and underscores")
    output = ROOT / "results" / f"{args.name}.json"
    if output.exists() or output.is_symlink():
        raise FileExistsError("Preserve prior results; choose a new summary name")
    metrics = {"kind": "t01v2_round_summary", "aggregation_status": "passed",
        "task_acceptance": None, "reported_counts_independently_verified": False,
        "shards": [], "needs_clone_next_round": [], "unknown_shards": [], "source_errors": []}
    config = {"name": args.name, "requested_sources": args.source,
        "count_scope": "Copy source-reported counts; no pair/journal revalidation or final-merger acceptance.",
        "trigger": "Each explicit ended first-round shard: four known counts, any class below 5.",
        "missing_policy": "Missing class counts remain unknown; never inferred from attempts or exit status.",
        "retry_policy": "Preserve each source independently, including failures; do not pool retries or choose a best run.",
        "execution_scope": "No job submission, budget decision, data selection, or simulator execution."}
    sources, seen_paths, seen_names = [], set(), set()
    for value in args.source:
        try:
            path = (ROOT / value).resolve()
            if path in seen_paths:
                raise ValueError("Duplicate explicit source path")
            seen_paths.add(path)
            data, reference = audited_result(path)
            row = summarize_source(data, reference)
            if row["run_name"] in seen_names:
                raise ValueError("Duplicate source run identity")
            seen_names.add(row["run_name"])
            sources.append(reference)
            metrics["shards"].append(row)
            entry = {key: row[key] for key in ("run_name", "scene", "pbs_jobid", "gpu_uuid")}
            entry["source_file"] = reference["file"]
            if row["needs_clone_next_round"] is True:
                metrics["needs_clone_next_round"].append({**entry, "below_threshold_classes": row["below_threshold_classes"]})
            elif row["needs_clone_next_round"] is None:
                metrics["unknown_shards"].append({**entry, "unknown_count_classes": row["unknown_count_classes"]})
        except Exception as error:
            metrics["source_errors"].append({"file": value, "error": f"{type(error).__name__}: {error}"})
    if metrics["source_errors"]:
        metrics["aggregation_status"] = "failed"
    metrics["all_explicit_sources_verified"] = not metrics["source_errors"]
    metrics["all_count_decisions_known"] = not metrics["source_errors"] and not metrics["unknown_shards"]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump({"metrics": metrics, "config": config, "sources": sources}, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print("FINAL_METRICS_JSON: " + json.dumps(metrics, sort_keys=True, allow_nan=False), flush=True)
    return 1 if metrics["source_errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Classify a hash-verified collect-only T01 journal after thresholds are fixed.

No simulator calls. Run through run_logged after committing code. The original
candidate journal and all exact measurement fields remain unchanged.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

from .pair_generator import classify, metrics_from_rows, check_configuration
from .thor_common import ROOT, write_result


def classify_row(raw, max_displacement_m, min_evidence_pixels):
    row = copy.deepcopy(raw)
    row["collection_status"] = raw["status"]
    if raw["status"] != "collected":
        return row
    physical = row.get("physical_checks", [])
    checked = []
    for check in physical:
        native = check.get("native_collision_clear")
        displacement = check.get("actual_edited_displacement_m")
        if native is None or displacement is None:
            checked.append(None)
        else:
            checked.append(native is True and check["pose_changes"]["inventory_exact"]
                           and not check["collateral_pose_changes"] and displacement <= max_displacement_m)
    valid = copy.deepcopy(row["validity"])
    valid["valid_physical"] = (all(checked) if len(checked) == 2 and all(v is not None for v in checked) else None)
    evidence = row.get("evidence_pixels_after_look")
    valid["evidence_available"] = evidence >= min_evidence_pixels if evidence is not None else None
    status, reasons, pending = classify(row["kind"], valid, row["prefix_comparison"])
    row.update(status=status, validity=valid, valid_physical=valid["valid_physical"],
               discard_reasons=reasons, low_diff_pending_review=pending,
               pending_has_other_failures=bool(pending and any(reason != "same_input_at_t" for reason in reasons)),
               threshold_classification="posthoc_explicit_thresholds_without_new_rendering")
    # Crucially no assignment to same_input_at_t / same_input_at_t_strict.
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-result", required=True, type=Path)
    parser.add_argument("--raw-attempts", required=True, type=Path)
    parser.add_argument("--max-displacement-m", type=float, required=True)
    parser.add_argument("--min-evidence-pixels", type=int, required=True)
    parser.add_argument("--name", default="pilot_pairs")
    args = parser.parse_args(argv)
    source = json.loads(args.source_result.read_text())
    meta = source.get("_meta", {})
    measured = source["metrics"]
    if meta.get("producer") != "scripts/run_logged.py" or meta.get("metrics_sha256") != hashlib.sha256(json.dumps(measured, sort_keys=True).encode()).hexdigest():
        raise ValueError("Invalid source result provenance")
    if source["config"].get("collect_only") is not True:
        raise ValueError("Source must be a collect-only run")
    raw_bytes = args.raw_attempts.read_bytes()
    raw_sha = hashlib.sha256(raw_bytes).hexdigest()
    expected = [item for item in measured.get("output_files", [])
                if (ROOT / item["path"]).resolve() == args.raw_attempts.resolve()]
    if len(expected) != 1 or expected[0]["sha256"] != raw_sha:
        raise ValueError("Journal hash/path does not match its logged collection result")
    config = copy.deepcopy(source["config"])
    config.update(name=args.name, collect_only=False, max_displacement_m=args.max_displacement_m,
                  min_evidence_pixels=args.min_evidence_pixels,
                  threshold_source="explicit post-collection CLI; raw data unchanged",
                  output_paths={"all": "data/world_pairs/pilot_classified_attempts.jsonl",
                                "passed": "data/world_pairs/pilot.jsonl",
                                "pending_review": "data/world_pairs/pilot_pending_review.jsonl"})
    check_configuration(config)
    paths = {key: ROOT / value for key, value in config["output_paths"].items()}
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("Classified pilot output already exists; original preserved")
    rows = [classify_row(json.loads(line), args.max_displacement_m, args.min_evidence_pixels)
            for line in raw_bytes.decode().splitlines() if line.strip()]
    ids = [row["attempt_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate attempt IDs in raw journal")
    for key, path in paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as stream:
            for row in rows:
                if key == "all" or row["status"] == key:
                    stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    metrics = metrics_from_rows(rows, config, measured.get("scene_failures", []), "classified_existing_observations")
    metrics["source"] = {"file": str(args.source_result),
        "file_sha256": hashlib.sha256(args.source_result.read_bytes()).hexdigest(),
        "metrics_sha256": meta["metrics_sha256"], "git_commit": meta.get("git_commit"),
        "exit_code": meta.get("exit_code"), "raw_journal": str(args.raw_attempts), "raw_journal_sha256": raw_sha}
    metrics["output_files"] = [{"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                               for path in paths.values()]
    write_result(args.name, metrics, config)
    return 0 if metrics["acceptance_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

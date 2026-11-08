"""Audit and merge explicitly supplied T01 collections/legacy results, CPU only.

Use run_logged after commit. Parent collection JSONs expand ONLY their recorded
scene runs, never a directory glob. --retry-of NEW_RESULT=OLD_RESULT authorizes
overlap between explicitly supplied attempts of the same logical partition;
all outcomes remain in the ledger and conflicting duplicate outcomes are flagged.
This command never writes an accepted pilot or changes a validity threshold.
"""
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import re
import shutil

from src.envs.thor_common import ROOT, write_result
from src.tools.t01_partition import (DEFAULT_MANIFEST, audited_result, candidate_identity, candidate_outcome,
                                    canonical_hash, load_manifest, sha_file)

KINDS = ("direct_flip", "direct_stay", "look_flip", "look_stay")


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
    metrics = {"kind": "t01_merged_collection", "aggregation_status": "failed", "acceptance_passed": False,
               "source_runs": [], "collection_runs": [], "missing_shard_results": []}
    config = {"name": args.name, "requested_sources": args.source, "retry_of": args.retry_of,
              "collect_only": True, "thresholds_applied": False,
              "notes": ["Completed aggregation is not T01 acceptance; no thresholds or strict fields are changed.",
                        "Every explicit failure remains in source_runs/collection_runs and every available journal row remains in scratch chunks.",
                        "Scene overlap is rejected except for explicitly linked retries within the same declared logical owner.",
                        "Duplicate physical pair proposals count once; conflicting measurements are reported separately, never resolved by best outcome.",
                        "Unknown pair identity is retained as an unidentified attempt, not counted as a unique candidate.",
                        "Review selection is first ten unique collected candidates per class in fixed numeric-scene/attempt order; failed checks remain visible.",
                        "PBS job IDs are listed for separate final accounting; no allocation usage or credit charge is inferred here."]}
    sources, cache, writer, groups = [], {}, None, {}
    errors, warnings = [], []
    code = 1
    try:
        manifest, manifest_ref = load_manifest(args.manifest)
        config["manifest"] = manifest_ref
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
                if (data["config"].get("manifest", {}).get("sha256") != manifest_ref["sha256"]
                        or declared != manifest["partitions"][partition]["scenes"]):
                    raise ValueError("Collection scene manifest differs from the declared partition")
                metrics["collection_runs"].append({"source": ref, "partition": partition,
                    "pbs_jobid": data["config"].get("pbs_jobid"), "status": data["metrics"].get("status"),
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
        scratch = Path(scratch_value).resolve() / "t01_merged" / args.name
        scratch.mkdir(parents=True, exist_ok=False)
        config["merged_scratch_directory"] = str(scratch)
        writer = JournalChunks(scratch)
        all_attempts, unidentified, leaf_paths = 0, 0, set()
        raw_counts = {kind: Counter() for kind in KINDS}
        raw_reasons = {kind: Counter() for kind in KINDS}
        observed_seconds = []
        for data, ref, origin, partition, scenes in leaves:
            if ref["file"] in leaf_paths:
                raise ValueError("Same leaf supplied both directly and through a collection")
            leaf_paths.add(ref["file"])
            cfg, measured = data["config"], data["metrics"]
            if cfg.get("collect_only") is not True or cfg.get("seed") != manifest["seed"]:
                raise ValueError("Only seed-zero collect-only evidence is supported")
            if partition == "legacy":
                if scenes != manifest["partitions"]["legacy"]["scenes"]:
                    raise ValueError("Legacy source must retain all six reserved scenes")
            else:
                shard = cfg.get("shard", {})
                if (len(scenes) != 1 or cfg.get("pilot_scenes") != scenes or shard.get("scene") != scenes[0]
                        or shard.get("partition") != partition or shard.get("manifest", {}).get("sha256") != manifest_ref["sha256"]):
                    raise ValueError("Shard metadata does not match its parent/manifest scene")
            record = {"source": ref, "origin": str(Path(origin).relative_to(ROOT)), "partition": partition,
                      "scenes": scenes, "pbs_jobid": cfg.get("runtime", {}).get("pbs_jobid") or cfg.get("shard", {}).get("pbs_jobid"),
                      "status": measured.get("status"), "recorded_attempts": measured.get("attempts"),
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
                    item = groups.setdefault(identity, {"occurrences": [], "kinds": set(), "statuses": set(), "outcomes": set(), "row": row})
                    item["occurrences"].append(occurrence)
                    item["kinds"].add(row["kind"])
                    item["statuses"].add(row.get("status"))
                    item["outcomes"].add(candidate_outcome(row))
            if measured.get("attempts") is not None and rows != measured["attempts"]:
                raise ValueError("Journal length disagrees with the logged attempt count")
            record["journal_rows_verified"] = rows

        counts = {kind: Counter() for kind in KINDS}
        conflicts, duplicates, reviews = [], [], []
        look_observed, look_strict = 0, 0
        low_diff = {kind: 0 for kind in KINDS if kind.startswith("look_")}
        for identity, item in groups.items():
            if len(item["occurrences"]) > 1:
                duplicates.append({"candidate_identity": identity, "occurrences": item["occurrences"]})
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
            if status == "collected":
                reviews.append((identity, item["row"]))
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
                       duplicate_groups=duplicates, conflicting_duplicate_groups=conflicts, review_selection=selected,
                       declared_scene_owners={scene: str(Path(origin).relative_to(ROOT)) for scene, origin in sorted(ownership.items())},
                       scenes_not_supplied=sorted({scene for group in manifest["partitions"].values() for scene in group["scenes"]} - set(ownership)),
                       pbs_job_ids=sorted({row["pbs_jobid"] for row in metrics["source_runs"] + metrics["collection_runs"] if row.get("pbs_jobid")}))
        metrics["aggregation_status"] = "completed" if not conflicts and not metrics["missing_shard_results"] else "incomplete_evidence"
        metrics["all_sixty_scenes_declared"] = not metrics["scenes_not_supplied"]
        metrics["all_supplied_scene_runs_succeeded"] = bool(metrics["source_runs"]) and all(
            row["source"]["exit_code"] == 0 for row in metrics["source_runs"] + metrics["collection_runs"])
        code = 0 if metrics["aggregation_status"] == "completed" else 1
    except (OSError, ValueError, KeyError, TypeError) as error:
        errors.append(f"{type(error).__name__}: {error}")
    finally:
        metrics["merged_journal_chunks"] = writer.close() if writer else []
        metrics["errors"], metrics["warnings"] = errors, warnings
        write_result(args.name, metrics, config)
        output.write_text(json.dumps({"metrics": metrics, "config": config, "sources": sources}, indent=2, allow_nan=False) + "\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())

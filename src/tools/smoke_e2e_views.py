"""E2E step 1: all five approved D4 exports plus exactly twenty fresh states.

Run through run_logged in a one-GPU PBS allocation. This isolated exception
does not change the formal T02 200-state check or authorize scientific training.
Large tables, raw actions and fresh captures remain in private scratch folders.
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import sys
import time
import traceback

from src.envs.thor_common import ROOT, Worker, deadline_after, install_signals, stop_workers
from src.envs.pair_table_export import load_table
from src.envs.pair_table_checks import check_pair, verify_render_sources, authenticate_cache_sources
from src.envs.pair_table_source import load_reviewed_pair
from src.tools.t01_partition import audited_result
from src.tools.t02_export import (VERSION as EXPORT_VERSION, save_json, execution_environment, gate_runtime,
    logged_gate, pair_worker, pair_runtime, preserve_black, black_only_failure)
from src.tools.t02_cache_check import fresh_worker, load_samples, read_json_ref
from src.tools.smoke_e2e_scope import VERSION, POLICY, digest, sample_plan, validate_sample, authorize_smoke, coverage

MODULE = "src.tools.smoke_e2e_views"
APPROVED_PAIRS = ("floorplan203_00038", "floorplan203_00039", "floorplan203_00041",
                  "floorplan203_00080", "floorplan203_00081")
APPROVED_CHECKPOINT = "results/t01v2_checkpoint_single_02.json"
APPROVED_SCENE = "results/pilot_pairs_v2_living_201_204_a00_r32d46927_r7b375c62_j707341_hopper_m_02_g0_floorplan203.json"


def file_ref(path, root, limit=48 * 1024 * 1024):
    path = Path(path).resolve()
    path.relative_to(Path(root).resolve())
    if not path.is_file() or path.stat().st_size > limit:
        raise ValueError("Missing/oversized smoke evidence file")
    body = path.read_bytes()
    return {"file": str(path), "sha256": hashlib.sha256(body).hexdigest(), "bytes": len(body)}


def review_packages(path):
    reference = file_ref(path, ROOT / "configs", 1024 * 1024)
    plan = json.loads(Path(reference["file"]).read_text())
    if (plan.get("version") != "smoke_e2e_reviewed_five_v1"
            or plan.get("checkpoint") != APPROVED_CHECKPOINT or plan.get("scene_source") != APPROVED_SCENE
            or plan.get("approval", {}).get("status") != "human_explicitly_approved"
            or tuple(row.get("pair_id") for row in plan.get("pairs", [])) != APPROVED_PAIRS
            or plan.get("planning", {}).get("gpus_per_PBS_job") != 1
            or plan["planning"].get("total_E2E_charged_GPU_hours_limit") != 4
            or plan["planning"].get("cache_distinct_pair_states") != 20):
        raise ValueError("Only the explicitly reviewed five-pair E2E plan is authorized")
    packages, image_refs, sources = [], [], []
    for index, row in enumerate(plan["pairs"]):
        kind = "direct_flip" if index < 3 else "direct_stay"
        image = f"logs/review/t01v2_checkpoint_single_02/{index:05d}_{kind}.png"
        if row.get("kind") != kind or row.get("review_image") != image:
            raise ValueError("Approved image/pair identity changed")
        image_refs.append(file_ref(ROOT / image, ROOT / "logs/review", 8 * 1024 * 1024))
        package = load_reviewed_pair(ROOT / APPROVED_CHECKPOINT, row["pair_id"])
        if (package["kind"] != kind
                or (ROOT / package["source_provenance"]["scene"]["file"]).resolve() != (ROOT / APPROVED_SCENE).resolve()):
            raise ValueError("Reviewed checkpoint points to another pair kind or scene source")
        packages.append(package)
        for source in package["sources"]:
            if source not in sources:
                sources.append(source)
    return plan, reference, image_refs, packages, sources


def worker_result(target, child):
    """Owned worker is cleaned before its output can authorize the next step."""
    worker = Worker(target, child)
    try:
        while worker.result is None and worker.error is None and time.time() < child["deadline"] + 1:
            worker.receive()
            time.sleep(.05)
        worker.receive()
        row = worker.result or {**worker.latest, "pair_id": child["package"]["pair_id"], "status": "error",
            "error": worker.error or {"type": "TimeoutError", "message": "Owned worker did not complete"},
            "cleanup_errors": []}
        row.update(worker_identity=worker.identity, unity_identity=worker.unity_identity,
                   scratch_directory=child["directory"])
    except BaseException:
        cleanup = stop_workers([worker])
        if cleanup:
            raise RuntimeError("Worker interrupted and cleanup uncertain: " + repr(cleanup))
        raise
    row["cleanup_errors"].extend(stop_workers([worker]))
    return row


def child_config(name, package, runtime, directory, deadline, pair_seconds, raw_bytes, table_bytes):
    return {"name": name, "scene": package["scene"], "resolution": 255,
            "runtime": pair_runtime(runtime, directory), "package": package, "directory": str(directory),
            "deadline": min(deadline - 15, time.time() + pair_seconds),
            "max_raw_bytes": raw_bytes, "max_table_bytes": table_bytes,
            **{key: package["source_config"][key] for key in ("grid_size", "visibility_distance", "field_of_view")}}


def completed(row, expected):
    return (row.get("status") == expected and row.get("cleanup_errors") == []
            and row.get("error") is None and not row.get("raw_index_error")
            and not row.get("black_frame_reference"))


def reload_export(row, package, directory):
    if (not completed(row, "exported") or row.get("pair_id") != package["pair_id"]
            or row.get("table_complete") is not True or row.get("native_evidence_verified") is not True
            or row.get("pair_checks_passed") is not True
            or Path(row["scratch_directory"]).resolve() != directory.resolve()):
        raise ValueError("Every selected pair must complete export checks; no subset filtering")
    stored = read_json_ref(row["source_package"], directory, expected=directory / "authenticated_source.json")
    for key in ("pair_id", "scene", "kind", "worlds", "t01", "initial_position", "initial_heading",
                "anchor_name", "candidate_names", "target_names", "instruction", "clone_context",
                "reachable_positions", "reachable_reference", "source_config"):
        if stored.get(key) != package.get(key):
            raise ValueError("Saved science input changed: " + key)
    for role, original in stored["source_provenance"].items():
        actual = package["source_provenance"][role]
        if any(original.get(key) != actual.get(key) for key in ("file", "sha256", "metrics_sha256", "git_commit")):
            raise ValueError("Original source bytes changed: " + role)
    details = read_json_ref(row["detail"], directory, expected=directory / "checks.json")
    manifest = row["table_manifest"]
    if (details["table_manifest"] != manifest
            or Path(manifest["file"]).resolve() != (directory / "table/table.json").resolve()):
        raise ValueError("Export manifest identity mismatch")
    read_json_ref(row["raw_index"], directory)
    table = load_table(manifest["file"], expected_sha256=manifest["sha256"])
    from src.envs.pair_table_export import FROZEN_RENDER_VERSION
    if (table.pair_id != package["pair_id"] or table.source.get("export_version") != EXPORT_VERSION
            or table.source.get("frozen_render_version") != FROZEN_RENDER_VERSION
            or table.source.get("render_source_worlds") != [world["reference"] for world in package["worlds"]]
            or table.source.get('candidate_policy') != package['source_config'].get('candidate_policy')
            or table.source.get("t01") != stored["source_provenance"]
            or any(getattr(table, key) != package[key] for key in ("scene", "kind", "instruction", "anchor_name"))
            or tuple(table.candidate_names) != tuple(package["candidate_names"])
            or tuple(table.target_names) != tuple(package["target_names"])):
        raise ValueError("Table is not the declared original pair")
    start = table.states[table.initial_state].view
    if table.positions[start.viewpoint_id] != package["initial_position"] or start.heading_index != package["initial_heading"]:
        raise ValueError("Table initial camera differs from the reviewed pair")
    roots = [*package["evidence_roots"], str(directory)]
    native = verify_render_sources(table, evidence_roots=roots)
    checks = check_pair(table, package["t01"])
    if (native.get("passed") is not True or checks.get("passed") is not True or table.synthetic
            or any(item.get("passed") is not True for item in checks.get("witness_replay", []))):
        raise ValueError("Independent T02 checks 1–4 failed or witness unknown")
    package = {**package, "cache_evidence_roots": roots, "cached_table_manifest": manifest}
    return table, package, checks


def verify_fresh(tables, packages, rows, directory, plan):
    groups = {}
    for pair, state in plan["selected_keys"]:
        groups.setdefault(pair, []).append(state)
    if [row.get("pair_id") for row in rows] != list(groups):
        raise ValueError("Fresh workers do not cover the exact declared sample groups")
    all_cached = {view.process_id for table in tables.values() for state in table.states.values() for view in state.captures}
    samples, authenticated = [], set()
    for index, (pair_id, state_ids) in enumerate(groups.items()):
        row, pair_dir = rows[index], directory / f"fresh_{index:04d}"
        if not completed(row, "captured") or Path(row["scratch_directory"]).resolve() != pair_dir.resolve():
            raise ValueError("Fresh worker failure/unknown cleanup blocks the whole smoke")
        read_json_ref(row["raw_index"], pair_dir)
        roots = [*packages[pair_id]["cache_evidence_roots"], str(pair_dir)]
        observed = load_samples(tables[pair_id], row, pair_dir, roots, state_ids)
        if len(observed) != len(state_ids) or any(view.process_id in all_cached for sample in observed for view in sample.fresh):
            raise ValueError("Fresh captures missing or process reused from cached export")
        authenticated.update(authenticate_cache_sources([tables[pair_id]], observed, evidence_roots=roots))
        samples.extend(observed)
    return samples, authenticated


def exact_command(command, name, review_plan):
    if not isinstance(command, list) or not command or not re.fullmatch(r"python(?:3(?:\.\d+)?)?", Path(command[0]).name):
        raise ValueError("Expected Python smoke producer")
    argv = command[1:]
    if argv[:1] == ["-u"]:
        argv = argv[1:]
    if argv[:2] != ["-m", MODULE]:
        raise ValueError("Not the dedicated E2E exporter")
    for key, value in (("--name", name), ("--review-plan", review_plan)):
        if argv.count(key) != 1 or argv.index(key) + 1 >= len(argv) or argv[argv.index(key) + 1] != value:
            raise ValueError("Producer argv/result identity mismatch")


def load_success(path):
    """CPU-only reauthentication before step 2; no saved pass flag is enough."""
    data, source = audited_result(path)
    config, metrics, meta = data["config"], data["metrics"], data["_meta"]
    name = config.get("name")
    exact_command(meta.get("command"), name, config.get("review_plan"))
    if (not re.fullmatch(r"smoke_e2e_[A-Za-z0-9_]+", str(name)) or meta.get("name") != name
            or meta.get("exit_code") != 0 or config.get("version") != VERSION or config.get("policy") != POLICY
            or config.get("gpus") != 1 or config.get("depth") != 4
            or metrics.get("kind") != "smoke_e2e_views" or metrics.get("step1_passed") is not True
            or metrics.get("cleanup_errors") != [] or metrics.get("errors") != []
            or metrics.get("producer_exit_code") != 0 or metrics.get("formal_T02_acceptance") is not None):
        raise ValueError("E2E step 1 is failed/incomplete or belongs to another contract")
    if not Path(config["scratch"]).is_absolute():
        raise ValueError("Step 1 scratch must be absolute")
    scratch = Path(config["scratch"]).resolve()
    directory = scratch / "smoke_e2e" / name / config["pbs_jobid"]
    if (Path(config["scratch_directory"]).resolve() != directory or not scratch.is_absolute()
            or not re.fullmatch(r"\d+\.hopper-m-\d+", config["pbs_jobid"])):
        raise ValueError("Step 1 scratch/PBS identity mismatch")
    plan, reference, images, packages, sources = review_packages(ROOT / config["review_plan"])
    if reference != config.get("review_plan_reference") or images != config.get("review_images"):
        raise ValueError("Reviewed plan/image bytes changed after export")
    gate, gate_ref = audited_result(metrics["gate"]["file"])
    if gate_ref != metrics["gate"]:
        # Audit annotations can vary; immutable provenance cannot.
        for key in ("file", "sha256", "metrics_sha256", "git_commit"):
            if gate_ref.get(key) != metrics["gate"].get(key):
                raise ValueError("Step 1 gate source changed")
    gate_runtime(gate, {key: config[key] for key in ("pbs_jobid", "cuda_visible_devices", "scratch", "cpu_affinity")},
                 config["node"], directory / "unity_logs")
    rows = metrics.get("exports", [])
    if [row.get("pair_id") for row in rows] != list(APPROVED_PAIRS):
        raise ValueError("Step 1 filtered the reviewed pair population")
    tables, checked_packages, pair_checks = {}, {}, {}
    for index, package in enumerate(packages):
        table, checked, checks = reload_export(rows[index], package, directory / f"export_{index:04d}")
        tables[table.pair_id], checked_packages[table.pair_id], pair_checks[table.pair_id] = table, checked, checks
    counts = {pair: checks["non_anchor_divergent_states"] for pair, checks in pair_checks.items()}
    if metrics.get("non_anchor_divergent_states_by_pair") != counts:
        raise ValueError("Saved Direct hidden-divergence counts differ from the actual table states")
    population = [(pair, state) for pair, table in tables.items() for state in table.states]
    sampling = read_json_ref(metrics["sampling_plan"], directory, expected=directory / "sampling_plan.json")
    if sampling != sample_plan(population):
        raise ValueError("Step 1 sampled another population")
    samples, identities = verify_fresh(tables, checked_packages, metrics["fresh_runs"], directory, sampling)
    report = validate_sample(samples, population=population, plan=sampling)
    saved = read_json_ref(metrics["cache_report"], directory, expected=directory / "smoke_cache.json")
    if digest(saved) != digest(report):
        raise ValueError("Cache report changed or is not reproduced by native source bytes")
    certificate = authorize_smoke(list(tables.values()), pair_checks=pair_checks, report=report, authenticated_ids=identities)
    saved_certificate = read_json_ref(metrics["smoke_certificate"], directory, expected=directory / "smoke_certificate.json")
    if certificate != saved_certificate:
        raise ValueError("Smoke certificate does not bind these exact tables and cache results")
    return tables, checked_packages, pair_checks, report, identities, certificate, [source, *sources, gate_ref]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--review-plan", required=True)
    parser.add_argument("--timeout-seconds", type=float, required=True)
    parser.add_argument("--pair-timeout-seconds", type=float, required=True)
    parser.add_argument("--gate-timeout-seconds", type=float, default=180)
    parser.add_argument("--max-raw-mib", type=int, default=2048)
    parser.add_argument("--max-table-mib", type=int, default=512)
    args = parser.parse_args()
    if (not re.fullmatch(r"smoke_e2e_[A-Za-z0-9_]+", args.name)
            or not all(math.isfinite(x) and x > 0 for x in (args.timeout_seconds, args.pair_timeout_seconds))
            or not 15 <= args.gate_timeout_seconds <= 600 or not args.gate_timeout_seconds + 60 < args.timeout_seconds <= 14340
            or args.pair_timeout_seconds > args.timeout_seconds - 30
            or not 1 <= args.max_raw_mib <= 2048 or not 1 <= args.max_table_mib <= 2048):
        parser.error("Unique smoke name and explicit bounded one-GPU E2E deadlines/caps required")
    output = ROOT / "results" / f"{args.name}.json"
    if output.exists() or output.is_symlink():
        raise FileExistsError("Preserve earlier smoke results; choose a unique name")
    config = {**vars(args), "version": VERSION, "policy": copy.deepcopy(POLICY), "gpus": 1,
              "depth": 4, "resolution": 255, "node": socket.gethostname(),
              "raw_cap_scope": "per owned export/fresh worker; not a whole-job cap",
              "budget_scope": "root must enforce cumulative 4 charged GPUh across ALL E2E jobs/retries"}
    metrics = {"kind": "smoke_e2e_views", "status": "incomplete", "step1_passed": False,
               "non_anchor_divergent_states_by_pair": {},
               "exports": [], "fresh_runs": [], "errors": [], "cleanup_errors": [], "gate": None,
               "black_frame_detected": False, "formal_T02_acceptance": None, "formal_cache_matches_live": None,
               "paper_result": False, "steps_3_to_5_started": False}
    sources, code = [], 1
    install_signals()
    try:
        identity = execution_environment(os.environ, os.sched_getaffinity(0))
        config.update(identity)
        deadline = deadline_after(args.timeout_seconds)
        config["absolute_deadline_epoch"] = deadline
        signal.setitimer(signal.ITIMER_REAL, max(.1, deadline - time.time()))
        directory = Path(identity["scratch"]) / "smoke_e2e" / args.name / identity["pbs_jobid"]
        directory.mkdir(parents=True, exist_ok=False)
        config["scratch_directory"] = str(directory)
        plan, reference, images, packages, sources = review_packages(ROOT / args.review_plan)
        config.update(review_plan_reference=reference, review_images=images, approved_pair_ids=list(APPROVED_PAIRS))
        gate, gate_ref, cleanup = logged_gate(args.name + "_gate", args.gate_timeout_seconds, directory, deadline, sources)
        metrics["gate"], metrics["cleanup_errors"] = gate_ref, cleanup
        if gate["_meta"]["exit_code"] != 0:
            metrics.update(status="blocked_by_render_gate", black_frame_detected=gate["_meta"]["exit_code"] == 86)
            code = 86 if metrics["black_frame_detected"] and not cleanup and gate["metrics"].get("cleanup_errors") == [] else 1
        else:
            runtime = gate_runtime(gate, identity, config["node"], directory / "unity_logs")
            config["runtime"] = runtime
            if cleanup:
                raise RuntimeError("Gate cleanup uncertain")
            tables, checked_packages, checks = {}, {}, {}
            for index, package in enumerate(packages):
                if time.time() >= deadline - 30:
                    raise TimeoutError("Deadline before next selected export")
                target = directory / f"export_{index:04d}"
                target.mkdir(exist_ok=False)
                package_ref = save_json(target / "authenticated_source.json", package)
                child = child_config(f"{args.name}_export{index}", package, runtime, target, deadline,
                    args.pair_timeout_seconds, args.max_raw_mib * 1024**2, args.max_table_mib * 1024**2)
                row = worker_result(pair_worker, child)
                row["source_package"] = package_ref
                metrics["exports"].append(row)
                metrics["cleanup_errors"].extend(row["cleanup_errors"])
                if row.get("black_frame_reference"):
                    row["portable_black_evidence"] = preserve_black(row["black_frame_reference"], target, runtime,
                        package["scene"], ROOT / "logs/runs" / args.name / f"black_export{index}")
                    metrics["black_frame_detected"] = True
                    code = 86 if black_only_failure(row) else 1
                    break
                table, checked, pair_checks = reload_export(row, package, target)
                tables[table.pair_id], checked_packages[table.pair_id], checks[table.pair_id] = table, checked, pair_checks
                metrics["non_anchor_divergent_states_by_pair"][table.pair_id] = pair_checks["non_anchor_divergent_states"]
            if not metrics["black_frame_detected"]:
                population = [(pair, state) for pair, table in tables.items() for state in table.states]
                sampling = sample_plan(population)
                metrics["sampling_plan"] = save_json(directory / "sampling_plan.json", sampling)
                groups = {}
                for pair, state in sampling["selected_keys"]:
                    groups.setdefault(pair, []).append(state)
                for index, (pair, states) in enumerate(groups.items()):
                    if time.time() >= deadline - 30:
                        raise TimeoutError("Deadline before next fresh sample group")
                    target = directory / f"fresh_{index:04d}"
                    target.mkdir(exist_ok=False)
                    package = checked_packages[pair]
                    child = child_config(f"{args.name}_fresh{index}", package, runtime, target, deadline,
                        args.pair_timeout_seconds, args.max_raw_mib * 1024**2, args.max_table_mib * 1024**2)
                    child["state_ids"] = states
                    row = worker_result(fresh_worker, child)
                    metrics["fresh_runs"].append(row)
                    metrics["cleanup_errors"].extend(row["cleanup_errors"])
                    if row.get("black_frame_reference"):
                        row["portable_black_evidence"] = preserve_black(row["black_frame_reference"], target, runtime,
                            package["scene"], ROOT / "logs/runs" / args.name / f"black_fresh{index}")
                        metrics["black_frame_detected"] = True
                        code = 86 if black_only_failure(row) else 1
                        break
                    if not completed(row, "captured"):
                        raise ValueError("Incomplete/failed fresh worker; stop the whole smoke")
                if not metrics["black_frame_detected"]:
                    samples, authenticated = verify_fresh(tables, checked_packages, metrics["fresh_runs"], directory, sampling)
                    report = validate_sample(samples, population=population, plan=sampling)
                    metrics["cache_report"] = save_json(directory / "smoke_cache.json", report)
                    certificate = authorize_smoke(list(tables.values()), pair_checks=checks, report=report, authenticated_ids=authenticated)
                    metrics["smoke_certificate"] = save_json(directory / "smoke_certificate.json", certificate)
                    metrics.update(step1_passed=True, status="smoke_step1_passed_step2_pending",
                                   observed_distinct_states=report["observed_distinct_states"], coverage=coverage(list(tables.values())))
                    code = 0
    except BaseException as error:
        traceback.print_exc()
        metrics["errors"].append(f"{type(error).__name__}: {error}")
        code = 1
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        if metrics["errors"] or metrics["cleanup_errors"]:
            code = 1
        if code:
            metrics["step1_passed"] = False
        metrics["producer_exit_code"] = code
        output.parent.mkdir(parents=True, exist_ok=True)
        save_json(output, {"metrics": metrics, "config": config, "sources": sources})
        print("FINAL_METRICS_JSON: " + json.dumps(metrics, sort_keys=True, allow_nan=False), flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())

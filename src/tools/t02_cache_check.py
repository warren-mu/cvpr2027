"""Formal >=200-state T02 cache check in one bounded, real PBS allocation.

Explicit completed export results define the entire population. Scientific
export rejections stay in its ledger; cache failures never remove a pair from
the population. No E2E exception, simulated live status, submission or training.
"""
import argparse
import ast
import copy
from contextlib import contextmanager
import hashlib
import io
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

from src.envs.thor_common import ROOT, Worker, controller_session, deadline_after, install_signals, stop_workers, process_identity
from src.envs.paired_env import RGBFrame
from src.envs.pair_table import RenderedView, eligible_names, label_assignment, verify_shadow_metadata
from src.envs.pair_table_export import load_table, SimulatorTableRenderer
from src.envs.pair_table_checks import (authenticate_cache_sources, authorize_tables, check_pair,
                                        verify_render_sources, verified_read)
from src.envs.pair_table_cache import (CACHE_POLICY, CacheSample, check_cached_state,
                                      sample_state_keys, validate_cache_sample)
from src.tools.t01_partition import audited_result, canonical_hash, sha_file
from src.tools.t02_export import (VERSION as EXPORT_VERSION, save_json, execution_environment,
                                  gate_runtime, logged_gate, preserve_black, black_only_failure, pair_runtime)

VERSION = "t02_formal_fresh_cache_dataset_v3"
INITIALIZE_RETRY_POLICY = {"version": "fresh_controller_initialize_timeout_retry_v1",
    "maximum_retries": 2, "same_pair_and_state_sequence": True, "deadline_extension": False,
    "cleanup_required": "original_owned_worker_stop_and_private_group_absent"}


def exact_export_command(command, name):
    if (not isinstance(command, list) or not command
            or not re.fullmatch(r"python(?:3(?:\.\d+)?)?", Path(command[0]).name)):
        raise ValueError("Expected an actual Python export producer")
    argv = command[1:]
    if argv[:1] == ["-u"]:
        argv = argv[1:]
    if (argv[:2] != ["-m", "src.tools.t02_export"] or argv.count("--name") != 1
            or argv.index("--name") + 1 >= len(argv) or argv[argv.index("--name") + 1] != name):
        raise ValueError("Source command is not this named frozen T02 exporter")


def read_json_ref(reference, root, *, expected=None):
    path = Path(reference["file"]).resolve()
    path.relative_to(Path(root).resolve())
    if expected is not None and path != Path(expected).resolve():
        raise ValueError("Reference is not its declared immutable export path")
    body = verified_read(reference, [root])
    if reference.get("bytes") is not None and reference["bytes"] != len(body):
        raise ValueError("Evidence size differs from recorded reference")
    return json.loads(body)


def load_exports(paths, scratch):
    """Recheck all original science, with one bounded evidence-byte snapshot."""
    from src.tools.t02_evidence_read_session import authenticated_read_reuse
    with authenticated_read_reuse(sys.modules[__name__]):
        return _load_exports(paths, scratch)


def _load_exports(paths, scratch):
    """Audit every declared export and recheck the complete usable inventory."""
    from src.envs.pair_table_source import load_reviewed_pair
    tables, packages, pair_checks, sources, ledger = {}, {}, {}, [], []
    seen_sources = set()
    for path in paths:
        data, reference = audited_result(path)
        if reference["file"] in seen_sources:
            raise ValueError("Duplicate explicit export source")
        seen_sources.add(reference["file"])
        sources.append(reference)
        meta, config, metrics = data["_meta"], data["config"], data["metrics"]
        name = config.get("name")
        exact_export_command(meta.get("command"), name)
        if (not re.fullmatch(r"t02_[A-Za-z0-9_]+", str(name)) or meta.get("name") != name
                or meta.get("exit_code") != 0 or config.get("version") != EXPORT_VERSION
                or config.get("gpus") != 1 or config.get("depth") != 4
                or metrics.get("kind") != "t02_table_export" or metrics.get("export_checks_complete") is not True
                or metrics.get("cleanup_errors") != [] or metrics.get("errors") != []
                or metrics.get("unstarted_pair_ids") != []):
            raise ValueError("Incomplete/failed export is not a complete population; retain it and stop")
        job = config.get("pbs_jobid", "")
        if not re.fullmatch(r"\d+\.hopper-m-\d+", job):
            raise ValueError("Export has no exact PBS identity")
        directory = Path(scratch).resolve() / "t02_export" / name / job
        if Path(config["scratch_directory"]).resolve() != directory:
            raise ValueError("Export root differs from its recorded private scratch directory")
        rows = metrics.get("pairs")
        if (not isinstance(rows, list) or [row.get("pair_id") for row in rows] != config.get("pair_id")
                or len(set(config["pair_id"])) != len(rows)):
            raise ValueError("Export did not account for every requested pair exactly once")
        for index, row in enumerate(rows):
            pair_id = row["pair_id"]
            pair_dir = directory / f"pair_{index:04d}"
            if Path(row["scratch_directory"]).resolve() != pair_dir or row.get("cleanup_errors") != []:
                raise ValueError("Pair ownership or cleanup is unknown")
            entry = {"pair_id": pair_id, "source": reference, "export_status": row["status"],
                     "export_rejection_reasons": copy.deepcopy(row.get("rejection_reasons")), "table_manifest": row.get("table_manifest")}
            ledger.append(entry)
            if row["status"] == "rejected":
                if not isinstance(row.get("rejection_reasons"), list) or not row["rejection_reasons"]:
                    raise ValueError("Export rejection has no explicit measured reason")
                entry["population_membership"] = "excluded_before_cache_by_export_checks"
                continue
            if (row["status"] != "exported" or row.get("table_complete") is not True
                    or row.get("native_evidence_verified") is not True or row.get("pair_checks_passed") is not True
                    or row.get("error") is not None or row.get("raw_index_error")):
                raise ValueError("An unknown/error pair cannot be silently excluded from cache validation")
            stored = read_json_ref(row["source_package"], pair_dir, expected=pair_dir / "authenticated_source.json")
            provenance = stored["source_provenance"]
            package = load_reviewed_pair(provenance["container"]["file"], pair_id,
                                         pair_file=provenance["pair"]["file"])
            if set(provenance) != set(package["source_provenance"]):
                raise ValueError("Source provenance roles changed")
            for role, original_ref in provenance.items():
                current_ref = package["source_provenance"][role]
                if any(original_ref.get(key) != current_ref.get(key)
                       for key in ("file", "sha256", "metrics_sha256", "git_commit")):
                    raise ValueError("Original T01 source bytes/provenance changed: " + role)
            # Fresh audit annotations may differ; the immutable science inputs
            # and their exact original source references may not.
            for key in ("pair_id", "scene", "kind", "worlds", "t01", "initial_position", "initial_heading",
                        "anchor_name", "candidate_names", "target_names", "instruction", "clone_context",
                        "reachable_positions", "reachable_reference", "source_config"):
                if stored.get(key) != package.get(key):
                    raise ValueError("Export source package differs from independently rechecked T01 inputs: " + key)
            details = read_json_ref(row["detail"], pair_dir, expected=pair_dir / "checks.json")
            manifest = row["table_manifest"]
            if details.get("table_manifest") != manifest or Path(manifest["file"]).resolve() != pair_dir / "table/table.json":
                raise ValueError("Pair and export-check record name different table bytes")
            table = load_table(manifest["file"], expected_sha256=manifest["sha256"])
            dataset_key = table.dataset_key
            if dataset_key in tables:
                raise ValueError('Duplicate pair source across exports; never choose a preferred retry')
            from src.envs.pair_table_export import FROZEN_RENDER_VERSION
            if (table.pair_id != pair_id or table.source.get("export_version") != EXPORT_VERSION
                    or table.source.get("frozen_render_version") != FROZEN_RENDER_VERSION
                    or table.source.get("render_source_worlds") != [world["reference"] for world in package["worlds"]]
                    or table.source.get('candidate_policy') != package['source_config'].get('candidate_policy')
                    or table.source.get('field_of_view', 100) != package['source_config'].get('field_of_view', 100)
                    or table.source.get("t01") != stored["source_provenance"]
                    or any(getattr(table, key) != package[key] for key in ("scene", "kind", "instruction", "anchor_name"))
                    or tuple(table.candidate_names) != tuple(package["candidate_names"])
                    or tuple(table.target_names) != tuple(package["target_names"])):
                raise ValueError("Stored table/source identity mismatch")
            start = table.states[table.initial_state].view
            if (table.positions[start.viewpoint_id] != package["initial_position"]
                    or start.heading_index != package["initial_heading"]):
                raise ValueError("Cached table changed the original initial viewpoint/heading")
            roots = [*package["evidence_roots"], str(pair_dir)]
            native = verify_render_sources(table, evidence_roots=roots)
            checks = check_pair(table, package["t01"])
            if native.get("passed") is not True or checks.get("passed") is not True or table.synthetic:
                raise ValueError("Previously usable export fails independent source checks")
            for declared in (row, details, details.get('pair_checks'), details.get('native_checks'),
                             details.get('export_measurements')):
                verify_shadow_metadata(table, declared)
            package["cache_evidence_roots"] = roots
            package["cached_table_manifest"] = manifest
            tables[dataset_key], packages[dataset_key], pair_checks[dataset_key] = table, package, checks
            entry["population_membership"] = "complete_independently_verified_table"
            entry.update(dataset_key=dataset_key, original_pair_id=pair_id,
                         dataset_identity=copy.deepcopy(table.dataset_identity))
            entry.update(table.shadow_metadata)
    if not tables:
        raise ValueError("No complete usable table in the explicit export population")
    return tables, packages, pair_checks, sources, ledger


def table_catalog(table):
    rows = [domain["evidence"] for state in table.states.values() for domain in state.graph_evidence]
    first = rows[0]
    if any(row["reachable_positions"] != first["reachable_positions"]
           or row["reachable_reference"] != first["reachable_reference"] for row in rows):
        raise ValueError("Cache renderer requires the export's single fixed actual native catalog")
    return copy.deepcopy(first["reachable_positions"]), copy.deepcopy(first["reachable_reference"])


def labels(view, table):
    assignment = label_assignment(table.pair_id, table.candidate_names)
    return {name: assignment[name] for name in eligible_names(view, table.candidate_names)}


def make_sample(table, state_id, fresh, *, use_dataset_key=False):
    cached = tuple(table.states[state_id].captures)
    return CacheSample(table.pair_id, state_id, table.anchor_name, tuple(table.candidate_names),
                       cached, tuple(fresh), tuple(labels(view, table) for view in cached),
                       tuple(labels(view, table) for view in fresh), kind=table.kind,
                       dataset_key=table.dataset_key if use_dataset_key else None)


def describe_view(view):
    return {"rgb_sha256": view.rgb.sha256,
            "mask_sha256": {name: hashlib.sha256(mask).hexdigest() for name, mask in view.masks.items()},
            "distances": copy.deepcopy(view.distances), "agent": copy.deepcopy(view.agent),
            "process_id": view.process_id, "render_id": view.render_id,
            "source": copy.deepcopy(view.source), "synthetic": view.synthetic}


def decode_view(description, roots):
    """Read actual PNG/NPZ bytes; no cached image or caller boolean can stand in."""
    import numpy as np
    from PIL import Image
    if description.get("synthetic") is not False:
        raise ValueError("Formal cache evidence must be real, not a synthetic fixture")
    capture = description["source"]["capture"]
    read = lambda reference: verified_read(reference, roots)
    metadata = json.loads(read(capture["files"]["metadata"]))
    names_in_scene = {obj["name"] for obj in metadata["objects"]}
    if len(names_in_scene) != len(metadata["objects"]):
        raise ValueError("Duplicate native object identities")
    with Image.open(io.BytesIO(read(capture["files"]["rgb"]))) as image:
        if image.mode != "RGB" or image.size != (255, 255):
            raise ValueError("Fresh raw RGB shape/type changed")
        rgb = RGBFrame(image.tobytes(), 255, 255)
    with np.load(io.BytesIO(read(capture["files"]["masks"])), allow_pickle=False) as masks:
        names, pixels = masks["names"].tolist(), masks["masks"]
        if len(names) != len(set(names)) or pixels.dtype != np.bool_ or pixels.shape != (len(names), 255, 255):
            raise ValueError("Fresh instance mask archive malformed")
        actual = {name: value.astype("uint8").tobytes() for name, value in zip(names, pixels)}
    desired = description["mask_sha256"]
    if not set(desired) <= names_in_scene:
        raise ValueError("Absent identity is unknown, not a zero instance mask")
    chosen = {name: actual.get(name, bytes(255 * 255)) for name in desired}
    if (rgb.sha256 != description["rgb_sha256"]
            or any(hashlib.sha256(mask).hexdigest() != desired[name] for name, mask in chosen.items())):
        raise ValueError("Fresh capture descriptor differs from native payload bytes")
    return RenderedView(rgb, chosen, description["distances"], description["agent"],
                        description["process_id"], description["render_id"], description["source"], False)


def initialize_timeout(error, scene):
    """Recognize only THOR's explicit Initialize action timeout, not deadlines."""
    if not isinstance(error, TimeoutError) or not isinstance(scene, str):
        return False
    match = re.fullmatch(r"Error encountered when running action (\{.*\}) in scene (.+)\.", str(error))
    if match is None or match.group(2) not in (scene, scene + "_physics"):
        return False
    try:
        action = ast.literal_eval(match.group(1))
    except (ValueError, SyntaxError):
        return False
    return isinstance(action, dict) and action.get("action") == "Initialize"


def initialization_cleanup_proof(worker, scope, cleanup):
    """Retain original group identity and require observed disappearance."""
    valid = (isinstance(scope, dict) and isinstance(worker.identity, dict)
             and scope.get("pid") == worker.process.pid
             and scope.get("start_ticks") == worker.identity.get("start_ticks")
             and scope.get("pgid") == scope.get("sid") == worker.process.pid)
    absent = False
    if valid:
        try:
            os.killpg(scope["pgid"], 0)
        except ProcessLookupError:
            absent = True
        except OSError:
            pass
    return {"scope": scope, "worker_exitcode": worker.process.exitcode,
            "cleanup_errors": list(cleanup), "private_group_absent": absent,
            "confirmed": valid and absent and worker.process.exitcode is not None and not cleanup}


def capture_fresh_pair(child):
    """At most two infra retries; every sampled state retains its original deadline."""
    attempts = []
    for attempt in range(INITIALIZE_RETRY_POLICY["maximum_retries"] + 1):
        config = copy.deepcopy(child)
        config["initialization_attempt"] = attempt
        config["name"] = child["name"] + f"_init{attempt}"
        config["runtime"] = pair_runtime(child["runtime"], Path(child["directory"]) / f"initialization_{attempt:02d}")
        worker, row, cleanup = None, None, []
        scope = None
        try:
            worker = Worker(fresh_worker, config)
            while worker.result is None and worker.error is None and time.time() < config["deadline"] + 1:
                worker.receive()
                time.sleep(.05)
            worker.receive()
            row = worker.result or {**worker.latest, "pair_id": child["package"]["pair_id"],
                "dataset_key": child["dataset_key"], "status": "error",
                "error": worker.error or {"type": "TimeoutError", "message": "Fresh worker did not complete"},
                "cleanup_errors": []}
            row.update(worker_identity=worker.identity, unity_identity=worker.unity_identity,
                       scratch_directory=child["directory"])
        finally:
            if worker is not None:
                scope = process_identity(worker.process.pid)
                cleanup = stop_workers([worker])
        row["cleanup_errors"].extend(cleanup)
        eligible = (row.get("initialize_timeout") is True and row.get("controller_initialized") is False
                    and row.get("states") == [] and row.get("status") == "error"
                    and row.get("error", {}).get("type") == "TimeoutError"
                    and initialize_timeout(TimeoutError(row["error"]["message"]), child["scene"]))
        proof = initialization_cleanup_proof(worker, scope, row["cleanup_errors"]) if eligible else None
        if eligible and not proof["confirmed"]:
            row["cleanup_errors"].append("Initialize retry stopped: owned Controller cleanup not confirmed")
        retry = bool(eligible and proof["confirmed"] and attempt < INITIALIZE_RETRY_POLICY["maximum_retries"]
                     and time.time() < child["deadline"] - 15)
        attempt_record = {"attempt": attempt, "report_reference": row.get("report_reference"),
            "worker_identity": row.get("worker_identity"), "unity_identity": row.get("unity_identity"),
            "requested_states": list(child["state_ids"]), "status": row["status"], "error": row.get("error"),
            "initialize_timeout": eligible, "cleanup_proof": proof, "retry_started_next": retry}
        attempt_record["record_reference"] = save_json(Path(child["directory"]) / f"initialization_record_{attempt:02d}.json", attempt_record)
        attempts.append(attempt_record)
        if not retry:
            row["initialization_attempts"] = attempts
            row["initialization_retry_policy"] = dict(INITIALIZE_RETRY_POLICY)
            return row
    raise AssertionError("Unreachable retry bound")


def fresh_worker(config, connection):
    from src.envs.pair_v2_simulator import BlackFrame, V2Simulator
    from src.envs.pair_v2_clone import V2CloneSimulator
    os.sched_setaffinity(0, config["runtime"]["cpu_affinity"])
    directory, package = Path(config["directory"]), config["package"]
    manifest = package["cached_table_manifest"]
    table = load_table(manifest["file"], expected_sha256=manifest["sha256"])
    use_dataset_key = config.get('dataset_key') is not None
    if use_dataset_key and config['dataset_key'] != table.dataset_key:
        raise ValueError('Fresh worker dataset key differs from its immutable table')
    report = {"pair_id": table.pair_id, "scene": table.scene, "status": "error", "states": [],
              "requested_states": config["state_ids"], "error": None, "cleanup_errors": [], "black_frame_reference": None,
              "controller_initialized": False, "initialize_timeout": False,
              "initialization_attempt": config.get("initialization_attempt", 0)}
    if use_dataset_key:
        report.update(dataset_key=table.dataset_key, original_pair_id=table.pair_id)
    sim, body_finished = None, False
    try:
        with controller_session(config, "controller", segmentation=True, connection=connection) as controller:
            report["controller_initialized"] = True
            try:
                cls = V2CloneSimulator if package["clone_context"] is not None else V2Simulator
                sim = cls(controller, config, directory / "raw")
                for snapshot in package["worlds"]:
                    _, restoration = sim.restore_snapshot(snapshot)
                    if restoration.get("restore_deterministic") is not True:
                        raise ValueError("source_world_restore_failed")
                reachable, reference = table_catalog(table)
                renderer = SimulatorTableRenderer(sim, package["worlds"], reachable, package["initial_position"],
                    table.anchor_name, table.candidate_names, pair_id=table.pair_id, reachable_reference=reference,
                    target_names=table.target_names, candidate_policy=table.source.get('candidate_policy'))
                for index, state_id in enumerate(config["state_ids"]):
                    sim.check_deadline()
                    fresh = renderer.render_pair(table.states[state_id].view)
                    descriptor = {"pair_id": table.pair_id, "state_id": state_id,
                                  "worlds": [describe_view(view) for view in fresh]}
                    ref = save_json(directory / f"state_{index:04d}.json", descriptor)
                    # Diagnostics happen only after both actual source captures
                    # were retained. Native authentication is independently run
                    # again in the parent before any live certificate is issued.
                    comparison = check_cached_state(make_sample(table, state_id, fresh, use_dataset_key=use_dataset_key))
                    diagnostic = save_json(directory / f"comparison_{index:04d}.json", comparison)
                    report["states"].append({"state_id": state_id, "capture_descriptor": ref,
                        "comparison": diagnostic, "checks_passed": comparison["checks_passed"],
                        "reasons": comparison["reasons"]})
                    connection.send(("progress", copy.deepcopy(report)))
                    if not comparison["checks_passed"]:
                        report["status"] = "cache_mismatch"
                        break
                else:
                    report["status"] = "captured"
            except BlackFrame as error:
                report.update(status="black_frame", black_frame_reference=error.reference)
            except BaseException as error:
                report.update(status="error", error={"type": type(error).__name__, "message": str(error)})
                traceback.print_exc()
            finally:
                body_finished = True
                if sim is not None:
                    try:
                        report["raw_index"] = sim.finish()
                    except BaseException as error:
                        report.update(status="error", raw_index_error=f"{type(error).__name__}: {error}")
    except BaseException as error:
        report.update(status="error", error={"type": type(error).__name__, "message": str(error)})
        if not report["controller_initialized"]:
            report["initialize_timeout"] = initialize_timeout(error, config.get("scene"))
        if body_finished:
            report["cleanup_errors"].append("Controller shutdown: " + str(error))
        traceback.print_exc()
    incremental = directory / "raw/evidence_index.jsonl"
    if incremental.is_file():
        report["raw_incremental_index"] = {"file": str(incremental), "sha256": sha_file(incremental),
                                            "bytes": incremental.stat().st_size}
    report_path = (directory / "worker_report.json" if report["controller_initialized"] else
                   directory / f"initialization_attempt_{report['initialization_attempt']:02d}.json")
    report["report_reference"] = save_json(report_path, report)
    return report


def load_samples(table, report, directory, roots, state_ids, *, use_dataset_key=False):
    if report.get("pair_id") != table.pair_id or report.get("requested_states") != state_ids:
        raise ValueError("Worker result belongs to another pair/sample plan")
    if use_dataset_key and (report.get('dataset_key') != table.dataset_key
                           or report.get('original_pair_id') != table.pair_id):
        raise ValueError('Worker result belongs to another dataset source')
    rows = report.get("states")
    if not isinstance(rows, list) or [row.get("state_id") for row in rows] != state_ids[:len(rows)]:
        raise ValueError("Fresh observations are not the declared state sequence")
    samples = []
    for index, row in enumerate(rows):
        descriptor = read_json_ref(row["capture_descriptor"], directory, expected=directory / f"state_{index:04d}.json")
        if descriptor.get("pair_id") != table.pair_id or descriptor.get("state_id") != row["state_id"] or len(descriptor.get("worlds", [])) != 2:
            raise ValueError("Fresh capture descriptor identity mismatch")
        fresh = tuple(decode_view(item, roots) for item in descriptor["worlds"])
        sample = make_sample(table, row["state_id"], fresh, use_dataset_key=use_dataset_key)
        actual = check_cached_state(sample)
        original = read_json_ref(row["comparison"], directory, expected=directory / f"comparison_{index:04d}.json")
        if canonical_hash(actual) != canonical_hash(original) or row.get("checks_passed") != actual["checks_passed"] or row.get("reasons") != actual["reasons"]:
            raise ValueError("Worker comparison differs from reconstructed actual bytes")
        samples.append(sample)
    return samples


@contextmanager
def engineering_phase(target, key):
    """Monotonic execution timings only; a normal return is not acceptance."""
    start, returned = time.perf_counter(), False
    try:
        yield
        returned = True
    finally:
        target[key] = {"elapsed_seconds": time.perf_counter() - start,
                       "returned_normally": returned}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--source", action="append", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=float, required=True)
    parser.add_argument("--pair-timeout-seconds", type=float, required=True)
    parser.add_argument("--gate-timeout-seconds", type=float, default=180)
    parser.add_argument("--max-raw-mib", type=int, default=2048)
    args = parser.parse_args()
    if (not re.fullmatch(r"t02_[A-Za-z0-9_]+", args.name) or args.sample_size < 200 or args.seed != 0
            or not all(math.isfinite(v) and v > 0 for v in (args.timeout_seconds, args.pair_timeout_seconds))
            or not 15 <= args.gate_timeout_seconds <= 600 or args.timeout_seconds <= args.gate_timeout_seconds + 60
            or args.pair_timeout_seconds > args.timeout_seconds - 30 or not 1 <= args.max_raw_mib <= 2048):
        parser.error("Formal cache requires >=200 states, registered seed 0, bounded deadlines and scratch caps")
    output = ROOT / "results" / f"{args.name}.json"
    if output.exists() or output.is_symlink():
        raise FileExistsError("Preserve prior cache checks; use a unique name")
    config = {**vars(args), "source": [str(p) for p in args.source], "version": VERSION,
              "policy": dict(CACHE_POLICY), "initialization_retry_policy": dict(INITIALIZE_RETRY_POLICY), "gpus": 1, "resolution": 255, "node": socket.gethostname(),
              "max_raw_mib_scope": "per owned fresh pair worker, not whole job",
              "sample_count_scope": "native-authenticated complete paired captures; unverified worker diagnostics retained separately",
              "engineering_timing_scope": "perf_counter wall durations; failed phases retained; normal return is not scientific acceptance; fresh validation includes final byte rehash"}
    metrics = {"kind": "t02_formal_cache_check", "status": "incomplete", "cache_matches_live": None,
               "downstream_allowed": False, "stop_downstream": True, "task_acceptance": None,
               "sampling_plan": None, "validation_report": None, "gate": None, "worker_runs": [],
               "population_ledger": [], "cleanup_errors": [], "errors": [], "black_frame_detected": False,
               "engineering_timings": {"version": "t02_engineering_phases_v1", "fresh_pairs": []}}
    sources, worker, samples, authenticated, code = [], None, [], set(), 1
    install_signals()
    try:
        identity = execution_environment(os.environ, os.sched_getaffinity(0))
        config.update(identity)
        deadline = deadline_after(args.timeout_seconds)
        config["absolute_deadline_epoch"] = deadline
        signal.setitimer(signal.ITIMER_REAL, max(.1, deadline - time.time()))
        directory = Path(identity["scratch"]) / "t02_cache" / args.name / identity["pbs_jobid"]
        directory.mkdir(parents=True, exist_ok=False)
        config["scratch_directory"] = str(directory)
        with engineering_phase(metrics["engineering_timings"], "gate"):
            gate, gate_ref, cleanup = logged_gate(args.name + "_gate", args.gate_timeout_seconds, directory, deadline, sources)
        metrics["gate"], metrics["cleanup_errors"] = gate_ref, cleanup
        if gate["_meta"]["exit_code"] != 0:
            metrics.update(status="blocked_by_render_gate", black_frame_detected=gate["_meta"]["exit_code"] == 86)
            code = 86 if metrics["black_frame_detected"] and not cleanup and gate["metrics"].get("cleanup_errors") == [] else 1
        else:
            runtime = gate_runtime(gate, identity, config["node"], directory / "unity_logs")
            config["runtime"] = runtime
            if cleanup:
                raise RuntimeError("Gate cleanup is uncertain")
            with engineering_phase(metrics["engineering_timings"], "load_exports"):
                tables, packages, pair_checks, export_sources, ledger = load_exports(args.source, identity["scratch"])
            sources.extend(export_sources)
            metrics["population_ledger"] = ledger
            metrics["table_shadow_metadata"] = {pair_id: table.shadow_metadata for pair_id, table in tables.items()}
            with engineering_phase(metrics["engineering_timings"], "population_sampling"):
                population = [(pair_id, state_id) for pair_id, table in tables.items() for state_id in table.states]
                plan = sample_state_keys(population, seed=args.seed, sample_size=args.sample_size)
                metrics["sampling_plan"] = save_json(directory / "sampling_plan.json", plan)
                config["population_sha256"] = plan["population_sha256"]
            # Group predetermined keys to amortize startup. This changes only
            # execution order; no key is added, removed, resampled or replaced.
            groups = {}
            for pair_id, state_id in plan["selected_keys"]:
                groups.setdefault(pair_id, []).append(state_id)
            all_cached_processes = {view.process_id for table in tables.values() for state in table.states.values() for view in state.captures}
            for index, (pair_id, state_ids) in enumerate(groups.items()):
                if time.time() >= deadline - 30:
                    metrics["errors"].append("cache_deadline_before_next_pair")
                    break
                package, pair_dir = packages[pair_id], directory / f"pair_{index:04d}"
                pair_dir.mkdir(exist_ok=False)
                child = {"name": f"{args.name}_p{index:04d}", "scene": package["scene"], "resolution": 255,
                    "dataset_key": pair_id,
                    "runtime": pair_runtime(runtime, pair_dir),
                    "package": package, "directory": str(pair_dir), "state_ids": state_ids,
                    "deadline": min(deadline - 15, time.time() + args.pair_timeout_seconds),
                    "max_raw_bytes": args.max_raw_mib * 1024 * 1024}
                for key in ("grid_size", "visibility_distance", "field_of_view"):
                    child[key] = package["source_config"][key]
                pair_timing = {"dataset_key": pair_id, "original_pair_id": package["pair_id"],
                               "sampled_state_count": len(state_ids)}
                metrics["engineering_timings"]["fresh_pairs"].append(pair_timing)
                with engineering_phase(pair_timing, "worker_execution_and_cleanup"):
                    row = capture_fresh_pair(child)
                metrics["worker_runs"].append(row)
                metrics["cleanup_errors"].extend(row["cleanup_errors"])
                if row.get("black_frame_reference"):
                    row["portable_black_evidence"] = preserve_black(row["black_frame_reference"], pair_dir,
                        runtime, package["scene"], ROOT / "logs/runs" / args.name / f"black_pair_{index:04d}")
                    metrics["black_frame_detected"] = True
                    code = 86 if black_only_failure(row) else 1
                    break
                if row["cleanup_errors"]:
                    break
                roots = [*package["cache_evidence_roots"], str(pair_dir)]
                from src.tools.t02_evidence_read_session import authenticated_read_reuse
                with engineering_phase(pair_timing, "fresh_validation_including_final_rehash"):
                    with authenticated_read_reuse(sys.modules[__name__]):
                        with engineering_phase(pair_timing, "decode_and_comparison"):
                            observed = load_samples(tables[pair_id], row, pair_dir, roots, state_ids, use_dataset_key=True)
                        if any(view.process_id in all_cached_processes for sample in observed for view in sample.fresh):
                            raise ValueError("Fresh process is also an export process in the complete population")
                        try:
                            with engineering_phase(pair_timing, "native_authentication"):
                                identities = authenticate_cache_sources([tables[pair_id]], observed, evidence_roots=roots)
                        except (ValueError, KeyError, TypeError, OSError) as error:
                            row["native_authentication_error"] = f"{type(error).__name__}: {error}"
                            metrics["errors"].append("fresh_native_provenance_failed:" + pair_id)
                            break
                samples.extend(observed)
                authenticated.update(identities)
                if row["status"] != "captured" or row.get("error") or row.get("raw_index_error"):
                    metrics["errors"].append("fresh_capture_incomplete_or_mismatched:" + pair_id)
                    break
            with engineering_phase(metrics["engineering_timings"], "sample_validation_and_authorization"):
                report = validate_cache_sample(samples, population=population, sampling_plan=plan)
                report["table_shadow_metadata"] = copy.deepcopy(metrics["table_shadow_metadata"])
                report["native_authenticated_render_ids"] = [list(key) for key in sorted(authenticated)]
                report["producer_cleanup_errors"] = list(metrics["cleanup_errors"])
                report["producer_errors"] = list(metrics["errors"])
                # A pure comparison result alone is never a real certificate.
                complete = (report["cache_matches_live"] is True and not metrics["cleanup_errors"]
                            and not metrics["errors"] and not metrics["black_frame_detected"])
                if complete:
                    authorize_tables(list(tables.values()), pair_checks=pair_checks, cache_report=report,
                                     authenticated_cache_source_ids=authenticated)
                    report["table_native_and_pair_checks_authorized"] = True
                else:
                    report.update(downstream_allowed=False, stop_downstream=True)
                metrics["validation_report"] = save_json(directory / "cache_validation.json", report)
                metrics.update(cache_matches_live=True if complete else False, downstream_allowed=complete,
                    stop_downstream=not complete, observed_distinct_states=report["observed_distinct_states"],
                    sample_complete=report["sample_complete"], missing_state_keys=report["missing_state_keys"],
                    status="cache_checks_passed_task_review_pending" if complete else "cache_failed_or_incomplete")
                if complete:
                    metrics["authorization_manifest"] = save_json(directory / "cache_authorization.json", {
                        "version": VERSION, "tables": {key: table.manifest_reference for key, table in tables.items()},
                        "table_shadow_metadata": copy.deepcopy(metrics["table_shadow_metadata"]),
                        "cache_validation": metrics["validation_report"], "sampling_plan": metrics["sampling_plan"],
                        "scope": "native export, pair checks and formal cache only; T02 balance/throughput/human review not certified",
                        "full_T02_acceptance": None})
                    code = 0
    except BaseException as error:
        traceback.print_exc()
        metrics["errors"].append(f"{type(error).__name__}: {error}")
        code = 1
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        if worker is not None:
            metrics["cleanup_errors"].extend(stop_workers([worker]))
        if metrics["cleanup_errors"] or metrics["errors"]:
            code = 1
        if code:
            metrics.update(downstream_allowed=False, stop_downstream=True)
        metrics["producer_exit_code"] = code
        output.parent.mkdir(parents=True, exist_ok=True)
        save_json(output, {"metrics": metrics, "config": config, "sources": sources})
        print("FINAL_METRICS_JSON: " + json.dumps(metrics, sort_keys=True, allow_nan=False), flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())

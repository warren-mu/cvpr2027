"""Independently verify explicit T01v2 shard evidence and publish the fixed pilot.

Run through run_logged after commit. No simulator is invoked. Selection is the
first 20 verified unique pairs per class in numeric-scene/journal/source order;
all source failures, rejected attempts and surplus qualified pairs remain in the
ledger. An ended failed run may contribute a fully verified completed pair only
when its provenance and cleanup are known. Incomplete/unowned evidence cannot.
"""
import argparse
from collections import Counter
import copy
import hashlib
import io
import json
import math
from pathlib import Path
import re
import shutil

from src.envs.thor_common import ROOT
from src.tools.t01_partition import audited_result, canonical_hash, sha_file

KINDS = ("direct_flip", "direct_stay", "look_flip", "look_stay")
PER_CLASS = 20
REVIEWS_PER_CLASS = 10


def finite_duration(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def reference_path(reference, *, root=ROOT, within=None):
    if (not isinstance(reference, dict) or not isinstance(reference.get("file"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", str(reference.get("sha256", "")))):
        raise ValueError("Artifact requires an explicit file and SHA-256")
    path = (Path(root) / reference["file"]).resolve()
    if within is not None and not path.is_relative_to(Path(within).resolve()):
        raise ValueError("Artifact escaped its declared evidence directory")
    if not path.is_file() or sha_file(path) != reference["sha256"]:
        raise ValueError("Missing or SHA-mismatched evidence: " + reference["file"])
    return path


def read_reference(reference, *, root=ROOT, within=None):
    return json.loads(reference_path(reference, root=root, within=within).read_text())


def source_eligibility(data):
    """Ended failure != unfinished source; preserve this distinction explicitly."""
    meta, metrics, config = data.get("_meta", {}), data.get("metrics", {}), data.get("config", {})
    reasons = []
    if type(meta.get("exit_code")) is not int or not meta.get("end"):
        reasons.append("source_not_formally_finished")
    if metrics.get("kind") != "pair_v2_shard" or config.get("seed") != 0 or config.get("resolution") != 255:
        reasons.append("wrong_source_kind_seed_or_resolution")
    if metrics.get("cleanup_errors") != []:
        reasons.append("source_cleanup_unknown_or_failed")
    if metrics.get("status") in (None, "running", "failed_before_generation", "worker_failed"):
        reasons.append("source_completion_or_worker_ownership_unknown")
    if not metrics.get("attempt_journal_sha256"):
        reasons.append("source_attempt_journal_not_observed")
    partial = meta.get("exit_code") != 0 or metrics.get("status") not in ("completed_search", "completed_quota")
    return {"eligible": not reasons, "partial_source": partial, "reasons": reasons,
            "exit_code": meta.get("exit_code"), "status": metrics.get("status"),
            "error": copy.deepcopy(metrics.get("error")), "cleanup_errors": copy.deepcopy(metrics.get("cleanup_errors"))}


def producer_arguments(command, module):
    """Match the executable position, never a module name buried in arguments."""
    if (not isinstance(command, list) or not command or not all(isinstance(x, str) for x in command)
            or re.fullmatch(r"python(?:[0-9]+(?:\.[0-9]+)*)?", Path(command[0]).name) is None):
        raise ValueError("Expected an explicit Python producer")
    position = 2 if len(command) > 1 and command[1] == "-u" else 1
    if command[position:position + 2] != ["-m", module]:
        raise ValueError("Logged command is not the required module producer")
    return command[position + 2:]


def shard_producer(command, config):
    """Clone evidence requires its separate, explicitly enabled producer."""
    enabled = config.get("enable_candidate_clone")
    version = config.get("preflight_version")
    if version is not None:
        from src.envs.pair_v2_stable import PREFLIGHT_VERSION, PREFLIGHT_POLICY
        if (version != PREFLIGHT_VERSION or enabled is not True or config.get("preflight_policy") != PREFLIGHT_POLICY
                or config.get("evidence_version") != "anchor_union_look_v1"):
            raise ValueError("Unknown stable preflight version or missing clone opt-in")
        module = "src.envs.pair_v2_stable_shard"
    elif enabled is True:
        module = "src.envs.pair_v2_clone_shard"
    elif enabled is None or enabled is False:
        module = "src.envs.pair_v2_shard"
    else:
        raise ValueError("Clone opt-in must be an explicit boolean")
    if version is None and config.get("evidence_version") is not None:
        raise ValueError("New evidence rule requires its explicit stable producer")
    return producer_arguments(command, module), enabled is True


def option(arguments, name):
    if arguments.count(name) != 1 or arguments.index(name) + 1 >= len(arguments):
        raise ValueError("Require one explicit argument " + name)
    return arguments[arguments.index(name) + 1]


def logged_configuration(text, prefix):
    rows = re.findall(r"^" + re.escape(prefix) + r":\s*(.+?)\s*$", text, re.M)
    if len(rows) != 1:
        raise ValueError("Expected one original " + prefix + " log record")
    return json.loads(rows[0])


def verify_source_identity(data, gate, *, logged_runtime=None, logged_allocation=None):
    """Bind the recorded shard to its actual runtime and audited gate allocation."""
    from src.tools.render_gate import gate_exit_code
    config, meta = data["config"], data["_meta"]
    shard, runtime = config["shard"], config.get("runtime")
    allocation = gate["config"]["allocation"]
    device = shard["gpu_identity"]
    logical = shard["logical_index"]
    devices = allocation["devices"]
    if type(logical) is not int or not 0 <= logical < len(devices):
        raise ValueError("Shard GPU logical index is outside the audited allocation")
    if (any(shard.get(key) != config.get(key) for key in ("name", "scene", "seed", "resolution"))
            or shard["name"] != meta["name"] or config.get("raw_evidence_dir") != shard.get("raw_evidence_dir")
            or shard.get("allocation") != allocation or logged_allocation != allocation
            or device != devices[logical] or device.get("logical_index") != logical
            or shard.get("pbs_jobid") != allocation.get("pbs_jobid")
            or gate["config"].get("pbs_jobid") != allocation.get("pbs_jobid")
            or shard.get("parent_cuda_visible_devices") != allocation.get("cuda_visible_devices")
            or shard.get("child_cuda_visible_devices") != device.get("gpu_uuid")
            or shard.get("node") != gate["config"].get("node")):
        raise ValueError("Shard identity/GPU/PBS allocation differs from its logged gate")
    cpus = shard.get("cpu_affinity")
    if (not isinstance(cpus, list) or len(cpus) != 12 or len(set(cpus)) != 12
            or any(type(cpu) is not int or cpu < 0 for cpu in cpus)
            or not set(cpus) <= set(allocation.get("cpu_affinity") or [])):
        raise ValueError("Shard must use twelve CPUs from the audited allocation affinity")
    if runtime is not None:
        if (runtime != logged_runtime or runtime.get("pbs_jobid") != shard["pbs_jobid"]
                or runtime.get("cuda_visible_devices") != device["gpu_uuid"]
                or runtime.get("cpu_affinity") != sorted(cpus)
                or any(runtime.get(key) != device.get(key) for key in
                       ("gpu_uuid", "gpu_index", "vulkan_device_index", "gpu_name", "mig_mode"))):
            raise ValueError("Shard runtime/log/GPU/CPU identity mismatch")
    elif logged_runtime is not None or source_eligibility(data)["eligible"]:
        raise ValueError("Contributing shard lacks its actual recorded runtime")
    gm, gc = gate["metrics"], gate["config"]
    rows = gm.get("devices") or []
    if (gate["_meta"].get("exit_code") != 0 or not gate["_meta"].get("end")
            or gm.get("passed") is not True or gm.get("cleanup_errors") != []
            or gm.get("errors") != [] or gate_exit_code(rows, len(devices), gm["errors"]) != 0
            or gc.get("resolution") != 255 or gc.get("segmentation") is not True
            or gm.get("expected_gpu_count") != len(devices) or gm.get("completed_gpu_count") != len(devices)):
        raise ValueError("Source gate is not a completed clean 255px segmentation gate")
    for index, row in enumerate(rows):
        observation = row["observation"]
        captures = observation.get("captures") or []
        if (observation.get("logical_index") != index or observation.get("gpu_identity") != devices[index]
                or observation.get("node") != shard["node"] or len(captures) != 2
                or any(c.get("segmentation", {}).get("valid") is not True
                       or c.get("shape") != [255, 255, 3] for c in captures)):
            raise ValueError("Gate rows do not verify each declared device and capture")


def journal_reference(data, eligibility):
    """Unavailable failed-source observations remain unknown, never zero."""
    metrics, config = data["metrics"], data["config"]
    declared = metrics.get("attempt_journal")
    if declared is not None and declared != config.get("attempt_journal"):
        raise ValueError("Source config/metrics journal disagreement")
    digest = metrics.get("attempt_journal_sha256")
    filename = declared or config.get("attempt_journal")
    if filename and digest:
        return {"file": filename, "sha256": digest}
    claims = metrics.get("passed_pair_files") or []
    counts = metrics.get("passed_by_class") or {}
    if (eligibility["eligible"] or metrics.get("status") in ("completed_search", "completed_quota")
            or claims or any(value not in (None, 0) for value in counts.values())
            or metrics.get("attempts") not in (None, 0)):
        raise ValueError("Source claims measured attempts/pairs without an authenticated journal")
    return None


def publication_eligible(counts, sources, source_errors, pair_errors):
    """Verified ledger-only failures do not veto other independently valid pairs."""
    ratio = counts["look_flip"] / counts["look_stay"] if counts["look_stay"] else None
    return (bool(sources) and not source_errors and not pair_errors
            and all(counts[k] >= PER_CLASS for k in KINDS) and ratio is not None and 0.8 <= ratio <= 1.25)


def load_source(path):
    data, ref = audited_result(path)  # P1-P8, plus mandatory exact FINAL/P6.
    meta, config = data["_meta"], data["config"]
    expected_name = Path(ref["file"]).stem
    if meta.get("name") != expected_name or config.get("name") != expected_name:
        raise ValueError("Source filename/meta/config name mismatch")
    if any("could not run git" in warning for warning in ref["audit"].get("warn", [])):
        raise ValueError("Source commit existence could not be established")
    arguments, clone_enabled = shard_producer(meta.get("command"), config)
    if (clone_enabled and data["metrics"].get("enable_candidate_clone") is not True
            or not clone_enabled and data["metrics"].get("enable_candidate_clone") is True):
        raise ValueError("Hashed metrics clone opt-in disagrees with the source producer/config")
    if data["metrics"].get("preflight_version") != config.get("preflight_version"):
        raise ValueError("Hashed metrics preflight version differs from source config")
    if data["metrics"].get("evidence_version") != config.get("evidence_version"):
        raise ValueError("Hashed metrics evidence rule differs from source config")
    if (option(arguments, "--name") != expected_name or option(arguments, "--scene") != config.get("scene")
            or data["metrics"].get("kind") != "pair_v2_shard" or not meta.get("end")
            or config.get("seed") != 0 or config.get("resolution") != 255):
        raise ValueError("Logged producer/name/scene does not identify a v2 shard")
    if config.get("metrics_spec_sha256") != sha_file(ROOT / "docs/METRICS_SPEC.md"):
        raise ValueError("Shard measured a different metrics specification")
    log = ROOT / meta.get("log_path", f"logs/runs/{expected_name}.log")
    side = ROOT / "logs/runs" / f"{expected_name}.meta.json"
    shard_file = (ROOT / option(arguments, "--shard-config")).resolve()
    if json.loads(shard_file.read_text()) != config["shard"]:
        raise ValueError("Immutable shard configuration differs from the recorded source config")
    gate, gate_ref = audited_result(config["shard"]["gate_result"])
    recorded_gate = config["shard"]["gate_source"]
    if any(recorded_gate.get(key) != gate_ref[key] for key in ("file", "sha256", "metrics_sha256", "exit_code", "git_commit")):
        raise ValueError("Shard's source gate hash/identity changed")
    gate_meta, gate_cfg = gate["_meta"], gate["config"]
    gate_args = producer_arguments(gate_meta.get("command"), "src.tools.render_gate")
    if (option(gate_args, "--name") != gate_cfg["name"] or gate_meta.get("name") != gate_cfg["name"]
            or Path(gate_ref["file"]).stem != gate_cfg["name"]
            or any("could not run git" in warning for warning in gate_ref["audit"].get("warn", []))):
        raise ValueError("Gate producer/name/commit identity mismatch")
    gate_log = ROOT / (gate_meta.get("log_path") or f"logs/runs/{gate_meta['name']}.log")
    log_text = log.read_text(errors="replace")
    runtime = logged_configuration(log_text, "RUNTIME_CONFIG") if config.get("runtime") is not None else None
    if runtime is None and re.search(r"^RUNTIME_CONFIG:", log_text, re.M):
        raise ValueError("Source config dropped its actual logged runtime")
    verify_source_identity(data, gate, logged_runtime=runtime,
                           logged_allocation=logged_configuration(gate_log.read_text(errors="replace"), "ALLOCATION_CONFIG"))
    ref.update(log_file=str(log.relative_to(ROOT)), log_sha256=sha_file(log),
               meta_file=str(side.relative_to(ROOT)), meta_sha256=sha_file(side), gate_source=gate_ref,
               shard_config_file=str(shard_file), shard_config_sha256=sha_file(shard_file),
               candidate_clone_enabled=clone_enabled)
    return data, ref


def scene_number(scene):
    match = re.fullmatch(r"FloorPlan([1-9][0-9]*)", str(scene))
    if not match:
        raise ValueError("Unknown scene identifier")
    return int(match.group(1))


def proposal_identity(row):
    """Deduplicate the declared construction, excluding file paths and outcomes."""
    identities, proposal, view = row["identities"], row["proposal"], row["view"]
    points = [{"receptacle_name": proposal[key]["spawn"]["receptacle_name"],
               "selected_position": proposal[key]["spawn"]["selected_position"]} for key in ("w0", "w1")]
    return canonical_hash({"scene": row["scene"], "identities": {
        "candidate_names": sorted(identities["candidate_names"]), "anchor_name": identities["anchor_name"]},
        "points": points, "view": {key: view[key] for key in ("position", "yaw", "horizon")}})


def verify_stable_preflight(preflight, read):
    """Recompute both phases from original action metadata and snapshot hashes."""
    from src.envs.pair_v2_stable import PREFLIGHT_VERSION, PREFLIGHT_POLICY, position_step, exceeded
    if (preflight.get("preflight_version") != PREFLIGHT_VERSION or preflight.get("policy") != PREFLIGHT_POLICY
            or preflight.get("passed") is not True or preflight.get("error") is not None
            or preflight.get("reasons") != []):
        raise ValueError("Stable preflight does not record a complete approved-protocol pass")
    physics = preflight["physics"]
    if physics != {"time_step_seconds": .02, "seconds_per_step": .1}:
        raise ValueError("Stable preflight changed the existing physics step duration")
    def snapshot(value):
        if read(value["reference"]) != {k: v for k, v in value.items() if k != "reference"}:
            raise ValueError("Stable preflight snapshot differs from its authenticated raw JSON")
        return value["objects"]
    def action(ref, name, before, arguments, previous_index=None):
        value = read(ref)
        if (value.get("action") != name or value.get("arguments") != arguments or value.get("exception") is not None
                or value["after_metadata"].get("lastActionSuccess") is not True
                or value["before_metadata"]["objects"] != before
                or type(value.get("index")) is not int
                or previous_index is not None and value["index"] != previous_index + 1):
            raise ValueError("Stable preflight action/metadata chain is inconsistent")
        return value
    diagnostics = preflight["diagnostics"]
    source = snapshot(diagnostics["source_snapshot"])
    pause = action(diagnostics["pause_action_reference"], "PausePhysicsAutoSim", source, {})
    if pause["index"] != 0:
        raise ValueError("Original-scene stable preflight was not the first backend operation")
    previous = pause["after_metadata"]["objects"]
    measured = position_step(source, previous, source)
    if measured != diagnostics["pause_measurements"] or not measured["inventory_exact"] or exceeded(measured, .5):
        raise ValueError("Initial pause exceeded the original-scene displacement/inventory bound")
    initial = diagnostics["initial_settling"]
    rows = initial["observations"]
    if not 1 <= len(rows) <= 50 or initial.get("settled") is not True:
        raise ValueError("Missing/beyond-N initial settling evidence")
    travelled = {n: v["observed_path_length_m"] for n, v in measured["objects"].items()}
    maxima = {n: v["baseline_displacement_m"] for n, v in measured["objects"].items()}
    streak, previous_index, criterion = 0, pause["index"], None
    args = {"timeStep": .02, "simSeconds": .1, "allowAutoSimulation": False}
    for index, row in enumerate(rows):
        value = action(row["action_reference"], "AdvancePhysicsStep", previous, args, previous_index)
        current = value["after_metadata"]["objects"]
        measured = position_step(previous, current, source, travelled)
        streak = streak + 1 if measured["all_steps_below_1mm"] else 0
        native = value["after_metadata"].get("isSceneAtRest")
        if (row["index"] != index or row["measurements"] != measured or row["small_step_streak"] != streak
                or row["native_is_scene_at_rest"] != native or not measured["inventory_exact"] or exceeded(measured, .5)):
            raise ValueError("Initial settling measurements do not recompute or exceed 0.5m")
        for name, item in measured["objects"].items(): maxima[name] = max(maxima[name], item["baseline_displacement_m"])
        previous, previous_index = current, value["index"]
        travelled = {n: v["observed_path_length_m"] for n, v in measured["objects"].items()}
        if native is True or streak >= 10:
            criterion = "native_rest" if native is True else "ten_small_steps"
            if index != len(rows) - 1: raise ValueError("Initial phase continued after its first settling criterion")
    baseline = snapshot(preflight["snapshot"])
    if (criterion is None or initial["criterion"] != criterion or baseline != previous
            or preflight["snapshot"].get("last_action_reference") != rows[-1]["action_reference"]):
        raise ValueError("Construction baseline is not the actual first settled snapshot")
    window = diagnostics["stability_window"]
    if window.get("completed") is not True or len(window["observations"]) != 50:
        raise ValueError("Stable window must contain exactly 50 measured steps without native-rest early stopping")
    travelled, window_maxima = {}, dict.fromkeys(maxima, 0.0)
    for index, row in enumerate(window["observations"]):
        value = action(row["action_reference"], "AdvancePhysicsStep", previous, args, previous_index)
        current = value["after_metadata"]["objects"]
        measured = position_step(previous, current, baseline, travelled)
        if (row["index"] != index or row["measurements"] != measured or not measured["inventory_exact"]
                or row["native_is_scene_at_rest"] != value["after_metadata"].get("isSceneAtRest") or exceeded(measured, .01)):
            raise ValueError("Stable window differs from raw metadata or exceeds its settled-baseline 1cm bound")
        for name, item in measured["objects"].items(): window_maxima[name] = max(window_maxima[name], item["baseline_displacement_m"])
        previous, previous_index = current, value["index"]
        travelled = {n: v["observed_path_length_m"] for n, v in measured["objects"].items()}
    if (snapshot(preflight["window_end_snapshot"]) != previous
            or preflight["window_end_snapshot"].get("last_action_reference") != window["observations"][-1]["action_reference"]):
        raise ValueError("Stable-window final snapshot is not the measured final state")
    return {"passed": True, "preflight_version": PREFLIGHT_VERSION, "initial_steps": len(rows),
            "settled_criterion": criterion, "window_steps": 50, "initial_max_displacement_m_by_name": maxima,
            "window_max_displacement_m_by_name": window_maxima, "baseline": "snapshot at initial settling; window_end kept separately"}


def verify_preflight(preflight, read=None):
    """Compare the original scene with its final rest; no extra transient bound."""
    if preflight.get("preflight_version") is not None:
        if read is None:
            raise ValueError("Versioned preflight requires authenticated native action evidence")
        return verify_stable_preflight(preflight, read)
    before = preflight["diagnostics"]["source_snapshot"]["objects"]
    after = preflight["snapshot"]["objects"]
    old, new = {o["name"]: o for o in before}, {o["name"]: o for o in after}
    if len(old) != len(before) or len(new) != len(after) or set(old) != set(new):
        raise ValueError("Original-scene preflight inventory changed")
    editable = {n for n, o in old.items() if o.get("moveable") or o.get("pickupable")}
    current = {n for n, o in new.items() if o.get("moveable") or o.get("pickupable")}
    if not editable or editable != current:
        raise ValueError("Preflight movable/pickupable inventory missing")
    deltas = {n: math.sqrt(sum((old[n]["position"][a] - new[n]["position"][a]) ** 2 for a in ("x", "y", "z")))
              for n in editable}
    settling = preflight["diagnostics"]["settling"]
    observations = settling.get("observations") or []
    if (preflight.get("passed") is not True or settling.get("settled") is not True
            or not observations or observations[-1].get("native_is_scene_at_rest") is not True
            or preflight["snapshot"].get("native_is_scene_at_rest") is not True
            or any(not math.isfinite(d) or d > 0.01 for d in deltas.values())):
        raise ValueError("Original scene did not settle within the approved 1cm final-position bound")
    return {"passed": True, "final_position_error_m_by_name": deltas,
            "scope": "original-to-final-rest positions; no additional transient or rotation threshold"}


def verify_clone_measurements(row, worlds, replays, restored, constructions, preflight, read):
    """Recompute clone validity from authenticated raw JSON, not saved PASS flags.

    ``read`` resolves an original file+SHA reference through the clone portable
    index. Original-scene preflight and the later enlarged baseline stay distinct.
    """
    from src.envs.pair_v2_clone_policy import (verify_clone_creation, clone_baseline_check,
                                             clone_pair_check, clone_replay_check)
    contexts = [world.get("clone_context") for world in worlds]
    if (len(contexts) != 2 or not isinstance(contexts[0], dict) or contexts[0] != contexts[1]
            or contexts[0].get("setup_passed") is not True or contexts[0].get("enabled") is not True):
        raise ValueError("Both worlds require the same explicit, verified clone baseline context")
    context = contexts[0]
    original = context["original_objects"]
    source_baseline = read(context["source_baseline_reference"])
    if (original != source_baseline["objects"] or original != preflight["snapshot"]["objects"]
            or context["source_baseline_reference"] != preflight["snapshot"]["reference"]
            or source_baseline != {k: v for k, v in preflight["snapshot"].items() if k != "reference"}
            or source_baseline.get("scene") != row["scene"] or source_baseline.get("clone_context") is not None):
        raise ValueError("Clone baseline is not derived from the separately verified original scene")
    recipe = context["recipe"]
    record = read(context["baseline_validation_reference"])
    creation = record["creation"]
    action = read(context["creation_action_reference"])
    if (record.get("scene") != row["scene"] or record.get("recipe") != recipe
            or record.get("original_snapshot_reference") != context["source_baseline_reference"]
            or record.get("passed") is not True or record.get("error") is not None
            or record.get("world_pair_edit") is not False
            or creation["action_reference"] != context["creation_action_reference"]
            or action.get("action") != "SetObjectPoses" or action.get("exception") is not None
            or action.get("arguments") != {"objectPoses": recipe["objectPoses"], "placeStationary": False}
            or action["after_metadata"].get("lastActionSuccess") is not True
            or creation.get("action_success") is not True
            or creation["created_objects"] != action["after_metadata"]["objects"]):
        raise ValueError("Native clone creation disagrees with its recipe and measured metadata")
    lineage = verify_clone_creation(original, creation["created_objects"], recipe, action_success=True)
    if not lineage["passed"] or context["lineage"] != lineage or creation["lineage"] != lineage:
        raise ValueError("Clone lineage cannot be independently reconstructed")
    if (sorted(row["identities"]["candidate_names"]) != sorted(lineage["candidate_names"])
            or row["identities"]["anchor_name"] in lineage["candidate_names"]):
        raise ValueError("The pair does not use the shared source/clone candidates and distinct anchor")
    physical = record["construction_physical"]
    clone_name = lineage["clone_name"]
    requests = physical["requested_poses"]
    clone_requests = [p for p in requests if p.get("objectName") == clone_name]
    if (len(clone_requests) != 1 or physical["before_objects"] != creation["created_objects"]
            or physical["edited_names"] != [clone_name]
            or set(physical["placement_evidence_by_name"]) != {clone_name}):
        raise ValueError("Common baseline must intentionally place only the actual newly created clone")
    proof = physical["placement_evidence_by_name"][clone_name]
    baseline = clone_baseline_check(original, creation["created_objects"], physical["realized_objects"], recipe,
        creation_action_success=True, requested_clone_pose=clone_requests[0], placement_evidence=proof,
        native_is_scene_at_rest=physical["native_is_scene_at_rest"])
    if not baseline["passed"] or record["clone_baseline_check"] != baseline:
        raise ValueError("Common clone baseline fails independent placement/rest/collateral checks")
    # The native duplicate creation can normalize Euler metadata. The subsequent
    # unique-name request preserves that measured created inventory, while the
    # independent baseline check above measures displacement from the ORIGINAL
    # scene. Do not turn raw Euler readback differences into a new hard gate.
    expected = {o["name"]: {"objectName": o["name"], "position": o["position"], "rotation": o["rotation"]}
                for o in creation["created_objects"] if o.get("moveable") or o.get("pickupable")}
    expected[clone_name] = clone_requests[0]
    actual = {p["objectName"]: p for p in requests}
    if len(actual) != len(requests) or actual != expected:
        raise ValueError("Clone physical request did not preserve the complete original inventory")
    request = read(physical["request_reference"])
    if request["complete_requested_poses"] != requests:
        raise ValueError("Clone requested poses changed after the pre-settling request record")
    settling = physical["settling"]
    if not settling.get("settled") or not settling.get("observations"):
        raise ValueError("Clone common baseline lacks actual settling observations")
    last = read(settling["observations"][-1]["action_reference"])
    if (last.get("action") != "AdvancePhysicsStep" or last.get("exception") is not None
            or last["after_metadata"].get("lastActionSuccess") is not True
            or last["after_metadata"].get("isSceneAtRest") is not True
            or last["after_metadata"]["objects"] != physical["realized_objects"]):
        raise ValueError("Clone final rest/positions disagree with the native physics action")
    geometry = proof["geometry"]
    query, placed = read(proof["query_reference"]), read(geometry["placement_reference"])
    returned = [o for o in placed["after_metadata"]["objects"] if o["name"] == clone_name]
    if (query.get("action") != "GetSpawnCoordinatesAboveReceptacle" or query.get("exception") is not None
            or query["after_metadata"].get("lastActionSuccess") is not True
            or query["after_metadata"].get("actionReturn") != proof["coordinates"]
            or placed.get("action") != "PlaceObjectAtPoint" or placed.get("exception") is not None
            or placed.get("arguments") != geometry["placement_arguments"]
            or placed["after_metadata"].get("lastActionSuccess") is not True or len(returned) != 1
            or returned[0]["position"] != geometry["returned_position"]
            or returned[0]["rotation"] != geometry["returned_rotation"]):
        raise ValueError("Clone legal pivot is not its recorded native pre-settling placement result")
    anchor = row["identities"]["anchor_name"]
    if constructions[0]["before_objects"] != physical["realized_objects"] or constructions[0]["edited_names"] != [anchor]:
        raise ValueError("w0 does not start at the shared physically validated clone baseline")
    baseline_poses = {o["name"]: {"objectName": o["name"], "position": o["position"], "rotation": o["rotation"]}
                      for o in physical["realized_objects"] if o.get("moveable") or o.get("pickupable")}
    w0_requests = {p["objectName"]: p for p in constructions[0]["requested_poses"]}
    if (len(w0_requests) != len(constructions[0]["requested_poses"]) or set(w0_requests) != set(baseline_poses)
            or any(w0_requests[n] != p for n, p in baseline_poses.items() if n != anchor)
            or w0_requests[anchor]["rotation"] != baseline_poses[anchor]["rotation"]):
        raise ValueError("w0 intentionally changed a shared candidate or another non-anchor object")
    paired = clone_pair_check(lineage, worlds[0]["objects"], worlds[1]["objects"],
                              worlds[0]["poses"], constructions[1]["requested_poses"], anchor)
    if paired["passed"] is not True or row.get("clone_pair_check") != paired:
        raise ValueError("Cloned candidates or the anchor-only counterfactual request are inconsistent")
    replay_checks = []
    for index in range(2):
        replay, snapshot = replays[index], restored[index]
        if snapshot.get("clone_context") != context or replay.get("clone_backend", {}).get("opt_in") is not True:
            raise ValueError("Fresh replay did not explicitly enable the same clone baseline")
        saved = read(snapshot["clone_recreation_reference"])
        if saved != replay["restoration"].get("clone_recreation") or saved.get("already_recreated"):
            raise ValueError("Fresh witness lacks a newly executed native clone recreation")
        recreated = read(saved["action_reference"])
        if (recreated.get("action") != "SetObjectPoses" or recreated.get("exception") is not None
                or recreated.get("arguments") != {"objectPoses": recipe["objectPoses"], "placeStationary": False}
                or recreated["after_metadata"].get("lastActionSuccess") is not True):
            raise ValueError("Fresh replay did not execute the saved complete duplicate recipe")
        checked = clone_replay_check(original, recreated["after_metadata"]["objects"], recipe, lineage, action_success=True)
        if not checked["passed"] or checked != {k: v for k, v in saved.items() if k != "action_reference"}:
            raise ValueError("Fresh replay actual inventory/name lineage differs from the saved shared baseline")
        replay_checks.append(checked)
    return {"passed": True, "lineage": lineage, "baseline": baseline, "pair": paired,
            "replay_recreation": replay_checks, "identity": {"recipe": recipe}}


def clone_evidence_reader(row, context, directory, *, root=ROOT, raw_root=None):
    """Authenticate the explicit clone JSON archive and its original-ref map."""
    evidence = row["clone_evidence"]
    archive = Path(directory).resolve().parent / "clone_evidence"
    if evidence.get("schema_version") != 1 or evidence.get("context") != context:
        raise ValueError("Clone archive context differs from both measured world snapshots")
    index = read_reference(evidence["index_reference"], root=root, within=archive)
    files = row.get("clone_evidence_files")
    if not isinstance(files, list) or not files or index.get("entries") != files:
        raise ValueError("Clone index and pair's explicit JSON file manifest differ")
    originals = {}
    for entry in files:
        source = entry.get("source_file")
        if not isinstance(source, str) or not Path(source).is_absolute():
            raise ValueError("Clone JSON proof requires its exact original absolute source path")
        source_path = Path(source).resolve()
        if raw_root is not None and not source_path.is_relative_to(Path(raw_root).resolve()):
            raise ValueError("Clone JSON original reference escaped its source evidence directory")
        path = reference_path(entry, root=root, within=archive)
        if path.suffix != ".json" or source_path.suffix != ".json" or source_path in originals:
            raise ValueError("Duplicate or non-JSON clone proof entry")
        originals[source_path] = (entry, path)

    def read(original):
        if (not isinstance(original, dict) or not isinstance(original.get("file"), str)
                or not Path(original["file"]).is_absolute()):
            raise ValueError("Clone proof needs an original absolute file reference")
        saved = originals.get(Path(original["file"]).resolve())
        if saved is None or saved[0]["sha256"] != original.get("sha256"):
            raise ValueError("Required original clone JSON proof is absent or has a different SHA")
        return json.loads(saved[1].read_text())

    return read, len(files)


def load_observation(record, portable, directory, *, root=ROOT):
    import numpy as np
    from PIL import Image
    result = {key: copy.deepcopy(record[key]) for key in
              ("metadata", "marks", "mark_labels_by_name", "input_text", "history")}
    for key in ("rgb", "marked", "masks"):
        ref = record["files"][key]
        path = reference_path(ref, root=root, within=directory)
        if portable.get(path) != ref["sha256"]:
            raise ValueError("Observation file absent from the portable manifest")
        if key != "masks":
            with Image.open(path) as image:
                if image.mode != "RGB" or image.size != (255, 255):
                    raise ValueError("Portable view is not original 255x255 RGB")
                result[key] = np.asarray(image).copy()
            if not np.any(result[key]):
                raise ValueError("Portable view is black")
        else:
            with np.load(path, allow_pickle=False) as archive:
                names = archive["names"].tolist()
                masks = archive["masks"]
                if (not isinstance(names, list) or not all(isinstance(n, str) for n in names)
                        or len(set(names)) != len(names) or masks.dtype != np.bool_
                        or masks.shape != (len(names), 255, 255)):
                    raise ValueError("Invalid portable stable-name mask archive")
                result["masks_by_name"] = {name: masks[index].copy() for index, name in enumerate(names)}
    return result


def check_pair(row, pair_path, *, root=ROOT, raw_root=None, clone_enabled=False, preflight_version=None, evidence_version=None):
    """Recompute the policy from portable measured inputs, never just trust PASS."""
    import numpy as np
    from src.envs.pair_v2_policy import (classify_pair, closest_target, compare_view,
        mark_observation, physical_check, prefix_check, public_copy, public_exact,
        restore_state_check, single_edit_check, witness_check)
    if preflight_version == "settled_baseline_window_v1":
        from src.envs.pair_v2_stable_policy import EVIDENCE_VERSION, look_evidence, classify_pair
        if evidence_version != EVIDENCE_VERSION:
            raise ValueError("Stable source lacks its explicit approved anchor-union evidence version")
    elif evidence_version is not None:
        raise ValueError("Historical sources cannot silently use the new evidence rule")
    directory = Path(pair_path).resolve().parent
    if row.get("status") != "passed" or row.get("kind") not in KINDS or row.get("discard_reasons") != []:
        raise ValueError("Pair was not recorded as an unqualified completed pass")
    files = row.get("portable_files")
    if not isinstance(files, list) or not files:
        raise ValueError("Missing portable evidence manifest")
    portable, by_name = {}, {}
    for ref in files:
        path = reference_path(ref, root=root, within=directory)
        if path in portable or path.name in by_name:
            raise ValueError("Duplicate portable artifact identity")
        portable[path], by_name[path.name] = ref["sha256"], ref

    def read(name):
        return read_reference(by_name[name], root=root, within=directory)

    worlds = [read(f"snapshot_w{i}.json") for i in range(2)]
    replays = [read(f"witness_w{i}.json") for i in range(2)]
    restored = [read(f"restored_snapshot_w{i}.json") for i in range(2)]
    constructions = [read(f"construction_w{i}.json") for i in range(2)]
    preflight_reference = row["scene_preflight_reference"]
    preflight_path = reference_path(preflight_reference, root=root, within=directory)
    if portable.get(preflight_path) != preflight_reference["sha256"]:
        raise ValueError("Original-scene preflight is absent from the portable manifest")
    raw_preflight = json.loads(preflight_path.read_text())
    if raw_preflight.get("preflight_version") != preflight_version:
        raise ValueError("Pair preflight version differs from its explicit source producer")
    preflight = verify_preflight(raw_preflight, lambda ref: read_reference(ref, root=root, within=raw_root))
    clone = None
    if (any(world.get("clone_context") is not None for world in worlds)
            or any(row.get(key) is not None for key in ("clone_evidence", "clone_pair_check", "clone_evidence_files"))):
        if clone_enabled is not True:
            raise ValueError("Clone pair cannot originate from the first-round natural-only producer")
        if row.get("enable_candidate_clone") is not True or row.get("candidate_origin") != "common_baseline_clone":
            raise ValueError("Clone pair lacks its explicit second-round candidate-origin declaration")
        reader, clone_file_count = clone_evidence_reader(row, worlds[0].get("clone_context"), directory,
                                                        root=root, raw_root=raw_root)
        clone = verify_clone_measurements(row, worlds, replays, restored, constructions, raw_preflight, reader)
        clone["json_proof_file_count"] = clone_file_count
    elif clone_enabled:
        original_names = {o["name"] for o in raw_preflight["snapshot"]["objects"]}
        if (row.get("enable_candidate_clone") is not True or row.get("candidate_origin") != "natural_fallback"
                or row.get("clone_selection", {}).get("status") != "natural_fallback"
                or any({o["name"] for o in world["objects"]} != original_names for world in worlds)):
            raise ValueError("Second-round natural fallback changed inventory or omitted its explicit declaration")
    record = read("observations.json")
    if set(record) != {"prefix", "after"} or any(len(record[key]) != 2 for key in record):
        raise ValueError("Require two actual prefix and two post-observation captures")
    prefix = [load_observation(r, portable, directory, root=root) for r in record["prefix"]]
    after = [load_observation(r, portable, directory, root=root) for r in record["after"]]
    identities = row["identities"]
    candidates, anchor = identities["candidate_names"], identities["anchor_name"]
    instruction = f"Select the {identities['target_type']} closest to the {identities['anchor_type']}."
    if row.get("instruction") != instruction:
        raise ValueError("Instruction does not match the recorded category-only task")
    for obs in (*prefix, *after):
        rebuilt = mark_observation(obs, candidates, instruction,
            public_context="Actions: move <index>; turn_left; turn_right; select <mark>.", history=())
        if not public_exact(obs, rebuilt) or obs["mark_labels_by_name"] != rebuilt["mark_labels_by_name"]:
            raise ValueError("Portable public input/marks differ from deterministic mask-derived input")
    for index, world in enumerate(worlds):
        if world.get("scene") != row["scene"] or prefix[index]["metadata"]["objects"] != world["objects"]:
            raise ValueError("Snapshot and captured prefix objects disagree")
        if constructions[index]["realized_objects"] != world["objects"]:
            raise ValueError("Physical evidence and saved world disagree")
        original = read_reference(world["reference"], root=root, within=raw_root)
        if original != {k: v for k, v in world.items() if k != "reference"}:
            raise ValueError("Portable snapshot changed from the native raw snapshot")
        restored_original = read_reference(replays[index]["restoration"]["restored_snapshot_reference"], root=root, within=raw_root)
        if restored_original != restored[index]:
            raise ValueError("Portable restored snapshot changed from the observed raw snapshot")
    target = [closest_target(w["objects"], candidates, anchor) for w in worlds]
    if [t.get("target_name") for t in target] != [row.get("target_w0"), row.get("target_w1")]:
        raise ValueError("Recorded targets differ from independently resolved targets")
    physical = [physical_check(c["before_objects"], c["requested_poses"], c["realized_objects"],
        c["edited_names"], native_is_scene_at_rest=c["native_is_scene_at_rest"],
        placement_evidence_by_name=c["placement_evidence_by_name"]) for c in constructions]
    if preflight_version == "settled_baseline_window_v1":
        from src.envs.pair_v2_stable import common_baseline_check, augment_physical_result
        common_objects = raw_preflight["snapshot"]["objects"]
        common_reference = raw_preflight["snapshot"]["reference"]
        if clone is not None:
            common_reference = worlds[0]["clone_context"]["baseline_validation_reference"]
            common_objects = reader(common_reference)["construction_physical"]["realized_objects"]
        for index, construction in enumerate(constructions):
            if construction["edited_names"] != [anchor]:
                raise ValueError("Stable worlds must intentionally edit only the distinct anchor")
            checked = common_baseline_check(common_objects, construction["realized_objects"], [anchor], common_reference)
            if (construction.get("common_baseline_check") != checked
                    or read_reference(construction["common_baseline_check_reference"], root=root, within=raw_root) != checked):
                raise ValueError("Stable-world common-baseline check differs from independently measured positions")
            physical[index] = augment_physical_result(physical[index], construction)
    for c in constructions:
        if c.get("settling", {}).get("settled") is not True or not c["settling"].get("observations"):
            raise ValueError("Missing actual bounded settling observations")
        if c["settling"]["observations"][-1].get("native_is_scene_at_rest") is not True:
            raise ValueError("Final physical simulation observation was not at rest")
        last = read_reference(c["settling"]["observations"][-1]["action_reference"], root=root, within=raw_root)
        if (last.get("action") != "AdvancePhysicsStep" or last.get("exception") is not None
                or last["after_metadata"].get("lastActionSuccess") is not True
                or last["after_metadata"].get("isSceneAtRest") is not True
                or last["after_metadata"]["objects"] != c["realized_objects"]):
            raise ValueError("Recorded rest/physical positions disagree with the actual final physics action")
        request = read_reference(c["request_reference"], root=root, within=raw_root)
        if request["complete_requested_poses"] != c["requested_poses"]:
            raise ValueError("Complete requested poses changed after construction")
        for name, proof in c["placement_evidence_by_name"].items():
            query = read_reference(proof["query_reference"], root=root, within=raw_root)
            geom = proof["geometry"]
            placement = read_reference(geom["placement_reference"], root=root, within=raw_root)
            if (query.get("action") != "GetSpawnCoordinatesAboveReceptacle" or query.get("exception") is not None
                    or query["after_metadata"].get("lastActionSuccess") is not True
                    or query["after_metadata"].get("actionReturn") != proof["coordinates"]
                    or placement.get("action") != "PlaceObjectAtPoint" or placement.get("exception") is not None
                    or placement["arguments"] != geom["placement_arguments"]
                    or placement["after_metadata"].get("lastActionSuccess") is not True):
                raise ValueError("Native placement evidence is inconsistent with its authenticated action")
            returned = [o for o in placement["after_metadata"]["objects"] if o["name"] == name]
            if len(returned) != 1 or returned[0]["position"] != geom["returned_position"]:
                raise ValueError("Requested pivot is not the recorded pre-settle native placement result")
    if constructions[1]["before_objects"] != worlds[0]["objects"] or constructions[1]["edited_names"] != [anchor]:
        raise ValueError("w1 was not constructed from w0 with only its anchor edited")
    single = single_edit_check(worlds[0]["poses"], constructions[1]["requested_poses"], anchor)
    look = row["kind"].startswith("look")
    shared0 = shared1 = None
    if look:
        shared_path = (Path(root) / row["shared_prefix_file"]).resolve()
        if shared_path != reference_path(record["prefix"][0]["files"]["marked"], root=root, within=directory):
            raise ValueError("Shared public prefix is not the authenticated w0 marked prefix")
        if row.get("shared_prefix_sha256") != portable.get(shared_path):
            raise ValueError("Shared prefix SHA mismatch")
        branches = row.get("shared_public_prefix_by_world")
        if not isinstance(branches, dict) or set(branches) != {"w0", "w1"} or branches["w0"] != branches["w1"]:
            raise ValueError("Both branches must reference the same complete public prefix artifact")
        public_path = reference_path(branches["w0"], root=root, within=directory)
        if portable.get(public_path) != branches["w0"]["sha256"]:
            raise ValueError("Shared public input lacks a portable manifest hash")
        public = json.loads(public_path.read_text())
        if set(public) != {"rgb", "marked", "input_text", "marks", "history"}:
            raise ValueError("Shared public input is incomplete or contains privileged fields")
        expected_public = {key: record["prefix"][0][key] for key in ("input_text", "marks", "history")}
        expected_public.update({key: record["prefix"][0]["files"][key] for key in ("rgb", "marked")})
        if public != expected_public:
            raise ValueError("Shared public input differs from the measured w0 public input")
        # Both branches use this one authenticated public input; branch-specific
        # original captures remain separate and are checked by D1 before reuse.
        shared0, shared1 = [public_copy(prefix[0])], [public_copy(prefix[0])]
    pre = prefix_check([prefix[0]], [prefix[1]], candidates, anchor,
        same_process=(type(worlds[0].get("worker_pid")) is int
            and worlds[0]["worker_pid"] == worlds[1].get("worker_pid")
            and type(worlds[0].get("unity_pid")) is int
            and worlds[0].get("unity_pid") == worlds[1].get("unity_pid")),
        agent_unchanged=worlds[0]["agent"] == worlds[1]["agent"],
        shared_prefix_w0=shared0, shared_prefix_w1=shared1)
    evidence_views = after if look else prefix
    evidence = (look_evidence(*evidence_views, anchor) if look and evidence_version == "anchor_union_look_v1"
                else compare_view(*evidence_views, [*candidates, anchor]))
    anchor_pixels = [int(obs["masks_by_name"].get(anchor, np.zeros((255, 255), dtype=bool)).sum()) for obs in evidence_views]
    restoration = [restore_state_check(w["objects"], r["objects"], w["agent"], r["agent"])
                   for w, r in zip(worlds, restored)]
    witnesses = []
    for index, replay in enumerate(replays):
        if (replay.get("status") != "completed" or replay.get("cleanup_errors") != []
                or replay.get("error") is not None or replay.get("construction_process_id") != worlds[index].get("worker_pid")
                or replay.get("process_id") != restored[index].get("worker_pid")):
            raise ValueError("Witness not complete, independently owned and cleanly stopped")
        if replay.get("restore_deterministic") != restoration[index]["restore_deterministic"]:
            raise ValueError("Witness restoration status disagrees with actual restored snapshot")
        checked = witness_check(replay, target[index]["target_name"])
        if replay.get("planned_actions") != row["witness_actions"][index]:
            raise ValueError("Saved witness public plan differs from pair record")
        if replay != row["witnesses"][index]["replay"]:
            raise ValueError("Portable witness differs from the pair's recorded executed witness")
        # Validate terminal measurements using the actual last replay capture,
        # without launching the simulator or guessing missing process evidence.
        terminal = replay["captures"][-1]["files"]
        meta = read_reference(terminal["metadata"], root=root, within=raw_root)
        mask_file = reference_path(terminal["masks"], root=root, within=raw_root)
        selected_name = replay["selected_name"]
        selected = [o for o in meta["objects"] if o["name"] == selected_name]
        if len(selected) != 1:
            raise ValueError("Terminal selected object absent/ambiguous in actual replay metadata")
        measured_distance = math.sqrt(sum((meta["agent"]["position"][axis] - selected[0]["position"][axis]) ** 2
                                         for axis in ("x", "y", "z")))
        with np.load(mask_file, allow_pickle=False) as archive:
            names = archive["names"].tolist()
            if (len(set(names)) != len(names) or selected_name not in names
                    or archive["masks"].dtype != np.bool_ or archive["masks"].shape != (len(names), 255, 255)):
                raise ValueError("Terminal selected object has no unique measured instance mask")
            pixels = int(archive["masks"][names.index(selected_name)].sum())
        reference_path(terminal["rgb"], root=root, within=raw_root)
        if measured_distance != replay["selected_distance_m"] or pixels != replay["selected_mask_pixels"]:
            raise ValueError("Terminal witness range/visibility differs from its raw observed capture")
        terminal_target = closest_target(meta["objects"], candidates, anchor)
        if not terminal_target["passed"] or terminal_target["target_name"] != selected_name:
            raise ValueError("Terminal replay does not select the actual unique closest target")
        witnesses.append(checked)
    common = replays[0]["executed_actions"][0] if look else None
    verdict = classify_pair(row["kind"], target_w0=target[0], target_w1=target[1], prefix=pre,
        evidence=evidence, physical_w0=physical[0], physical_w1=physical[1], single_edit=single,
        witness_w0=witnesses[0], witness_w1=witnesses[1],
        restore_deterministic=all(r["restore_deterministic"] for r in restoration),
        restore_max_abs_diff=[r.get("restore_max_abs_diff") for r in replays],
        anchor_pixels_w0=anchor_pixels[0], anchor_pixels_w1=anchor_pixels[1], common_observation_action=common)
    if verdict["acceptance_passed"] is not True or row.get("validity", {}).get("acceptance_passed") is not True:
        raise ValueError("Independent pair acceptance failed: " + repr(verdict["reasons"]))
    review = row["review_image"]
    review_path = reference_path(review, root=root, within=directory)
    if portable.get(review_path) != review["sha256"] or review_path.name != "review.png":
        raise ValueError("Review image is not uniquely authenticated portable evidence")
    identity = proposal_identity(row)
    if clone is not None:
        identity = canonical_hash({"pair_proposal": identity, "clone_baseline": clone["identity"]})
    return {"validity": verdict, "prefix": pre, "evidence": evidence, "physical": physical,
            "scene_preflight": preflight, "restoration": restoration,
            "clone_verification": clone,
            "witnesses": [{"passed": w["passed"], "target_name": w["target_name"]} for w in witnesses],
            "portable_file_count": len(portable), "review_image": review,
            "proposal_identity": identity}


def balanced_selection(qualified):
    ordered = sorted(qualified, key=lambda item: (scene_number(item["scene"]), item["journal_index"], item["source_file"]))
    selected, surplus, duplicates = [], [], []
    counts, seen = Counter(), {}
    for item in ordered:
        identity = item["proposal_identity"]
        if identity in seen:
            old = seen[identity]
            if (old["kind"], old["target_w0"], old["target_w1"]) != (item["kind"], item["target_w0"], item["target_w1"]):
                raise ValueError("Duplicate construction has conflicting class/targets")
            duplicates.append(item)
            continue
        seen[identity] = item
        if counts[item["kind"]] < PER_CLASS:
            selected.append(item)
            counts[item["kind"]] += 1
        else:
            surplus.append(item)
    return selected, surplus, duplicates


def durations(values):
    measured = [value for value in values if finite_duration(value)]
    return {"observed_count": len(measured), "missing_count": len(values) - len(measured),
            "mean_seconds": sum(measured) / len(measured) if measured else None,
            "total_observed_seconds": sum(measured) if measured else None}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="pilot_pairs_v2", choices=("pilot_pairs_v2",))
    parser.add_argument("--source", nargs="+", action="extend", required=True)
    args = parser.parse_args(argv)
    output = ROOT / "data/world_pairs/pilot_pairs_v2.jsonl"
    review_dir = ROOT / "logs/review/pilot_v2"
    result_path = ROOT / "results" / f"{args.name}.json"
    if any(path.exists() for path in (output, review_dir, result_path)):
        raise FileExistsError("Existing pilot/result/review evidence is preserved; no overwrite")
    metrics = {"kind": "pair_v2_merged_pilot", "status": "failed", "acceptance_passed": False,
        "passed_by_class": {k: 0 for k in KINDS}, "source_runs": [], "rejection_counts": {},
        "source_errors": [], "pair_errors": [], "scenes_excluded": [], "review_files": []}
    config = {"name": args.name, "requested_sources": args.source, "selection_per_class": PER_CLASS,
        "review_per_class": REVIEWS_PER_CLASS, "order": "numeric scene, journal row, source filename",
        "partial_source_policy": "formally ended failed source may supply independently complete pairs only with known clean cleanup and verified provenance",
        "ledger_policy": "audited ended noncontributing failures remain in the ledger; missing journal/counts are unknown; provenance contradictions block publication",
        "evidence_retention": "native action/snapshot references and immutable shard configs still require the original scratch files; no scratch-independent archival claim",
        "metrics_spec_sha256": sha_file(ROOT / "docs/METRICS_SPEC.md"),
        "dataset": str(output.relative_to(ROOT)), "review_directory": str(review_dir.relative_to(ROOT))}
    sources, qualified, attempts, reasons = [], [], [], Counter()
    code = 1
    try:
        paths = [(ROOT / path).resolve() for path in args.source]
        if len(paths) != len(set(paths)):
            raise ValueError("Duplicate explicit source")
        for path in paths:
            try:
                data, reference = load_source(path)
                sources.append(reference)
                measured, cfg = data["metrics"], data["config"]
                eligible = source_eligibility(data)
                source = {"source": reference, **eligible, "scene": cfg["scene"],
                          "pbs_jobid": cfg["shard"]["pbs_jobid"], "recorded_attempts": measured.get("attempts"),
                          "recorded_passed_by_class": measured.get("passed_by_class"),
                          "recorded_rejection_counts": measured.get("discard_reason_counts"),
                          "ledger_only": not eligible["eligible"], "verified_attempts": None,
                          "attempt_journal_status": "unknown"}
                metrics["source_runs"].append(source)
                scene_number(cfg["scene"])
                if measured.get("scene_excluded") is not None:
                    metrics["scenes_excluded"].append({"source": reference["file"], **measured["scene_excluded"]})
                journal_ref = journal_reference(data, eligible)
                if journal_ref is None:
                    source["attempt_journal"] = None
                    continue
                journal = reference_path(journal_ref, within=ROOT / "data/world_pairs/v2")
                source["attempt_journal"] = journal_ref
                passed_refs = measured.get("passed_pair_files")
                if passed_refs is None and not eligible["eligible"]:
                    passed_refs = []  # Validated against the journal below; not an inferred source metric.
                if not isinstance(passed_refs, list):
                    raise ValueError("Missing recorded passed-pair references")
                by_path = {}
                for item in passed_refs:
                    verified = reference_path(item, within=ROOT / "data/world_pairs/v2" / cfg["name"])
                    if verified in by_path:
                        raise ValueError("Duplicate recorded passed-pair path")
                    by_path[verified] = item
                seen, matched, rows = set(), set(), []
                for line in journal.read_text().splitlines():
                    if not line.strip():
                        continue
                    compact = json.loads(line)
                    if compact.get("scene") != cfg["scene"] or compact.get("pair_id") in seen:
                        raise ValueError("Wrong-scene or duplicated attempt identity")
                    seen.add(compact["pair_id"])
                    raw = read_reference(compact["raw_attempt"], within=cfg["raw_evidence_dir"])
                    if any(raw.get(key) != compact.get(key) for key in
                           ("pair_id", "scene", "kind", "status", "discard_reasons", "generation_seconds", "pair_file", "pair_sha256")):
                        raise ValueError("Journal and authenticated raw attempt disagree")
                    attempts.append({"source_file": reference["file"], "journal_index": len(rows), **compact})
                    rows.append(compact)
                    reasons.update(compact.get("discard_reasons") or [])
                    if compact.get("status") != "passed":
                        continue
                    pair_path = (ROOT / compact["pair_file"]).resolve()
                    if pair_path not in by_path:
                        raise ValueError("Passed journal row lacks a logged pair-file SHA")
                    if compact.get("pair_sha256") != by_path[pair_path]["sha256"]:
                        raise ValueError("Passed journal/pair hash disagree")
                    matched.add(pair_path)
                    if not eligible["eligible"]:
                        reasons["source_ineligible_completed_pair_not_admitted"] += 1
                        continue
                    pair = read_reference(by_path[pair_path], within=pair_path.parent)
                    source_preflight = measured["scene_preflight_reference"]
                    reference_path(source_preflight, within=cfg["raw_evidence_dir"])
                    if pair["scene_preflight_reference"]["sha256"] != source_preflight["sha256"]:
                        raise ValueError("Pair's portable preflight differs from the source's logged preflight")
                    if any(pair.get(key) != compact.get(key) for key in ("pair_id", "scene", "kind", "status", "discard_reasons")):
                        raise ValueError("Portable pair disagrees with journal identity/outcome")
                    try:
                        checked = check_pair(pair, pair_path, raw_root=cfg["raw_evidence_dir"],
                                             clone_enabled=cfg.get("enable_candidate_clone") is True,
                                             preflight_version=cfg.get("preflight_version"),
                                             evidence_version=cfg.get("evidence_version"))
                        qualified.append({"pair_id": pair["pair_id"], "scene": pair["scene"], "kind": pair["kind"],
                            "target_w0": pair["target_w0"], "target_w1": pair["target_w1"],
                            "source_file": reference["file"], "source_metrics_sha256": reference["metrics_sha256"],
                            "partial_source": eligible["partial_source"], "journal_index": len(rows) - 1,
                            "generation_seconds": compact.get("generation_seconds"), "pair_reference": by_path[pair_path],
                            "review_image": checked["review_image"], "proposal_identity": checked["proposal_identity"],
                            "independent_verification": checked, "pair": pair})
                    except Exception as error:
                        metrics["pair_errors"].append({"source": reference["file"], "pair_id": compact["pair_id"], "error": str(error)})
                        reasons["independent_pair_verification_failed"] += 1
                if ((measured.get("attempts") is not None and len(rows) != measured["attempts"])
                        or (eligible["eligible"] and measured.get("attempts") is None) or matched != set(by_path)):
                    raise ValueError("Journal count or passed-pair references disagree with logged metrics")
                counted = {kind: sum(r.get("kind") == kind and r.get("status") == "passed" for r in rows) for kind in KINDS}
                if ((measured.get("passed_by_class") is not None and counted != measured["passed_by_class"])
                        or (eligible["eligible"] and measured.get("passed_by_class") is None)):
                    raise ValueError("Logged per-class counts disagree with the journal")
                source.update(verified_attempts=len(rows), attempt_journal_status="verified")
            except Exception as error:
                metrics["source_errors"].append({"file": str(path), "error": f"{type(error).__name__}: {error}"})
        selected, surplus, duplicates = balanced_selection(qualified)
        counts = {kind: sum(item["kind"] == kind for item in selected) for kind in KINDS}
        ratio = counts["look_flip"] / counts["look_stay"] if counts["look_stay"] else None
        acceptance = publication_eligible(counts, sources, metrics["source_errors"], metrics["pair_errors"])
        def compact(item):
            return {k: v for k, v in item.items() if k not in ("pair", "independent_verification")}
        metrics.update(passed_by_class=counts, qualified_by_class={k: sum(r["kind"] == k for r in qualified) for k in KINDS},
            selected_pairs=[compact(r) for r in selected], qualified_not_selected=[compact(r) for r in surplus],
            duplicate_completed_pairs=[compact(r) for r in duplicates], look_flip_stay_ratio=ratio,
            attempts=(len(attempts) if not metrics["source_errors"] and all(
                r["verified_attempts"] is not None for r in metrics["source_runs"]) else None),
            observed_attempts=len(attempts), unknown_attempt_source_count=sum(
                r["verified_attempts"] is None for r in metrics["source_runs"]),
            rejection_counts=dict(reasons), scene_exclusion_count=len(metrics["scenes_excluded"]),
            attempt_generation_time=durations([r.get("generation_seconds") for r in attempts]),
            accepted_generation_time=durations([r.get("generation_seconds") for r in selected]),
            timing_scope="recorded construction durations only; no PBS wait/allocation or unmeasured setup attributed",
            source_failure_count=sum(row["partial_source"] for row in metrics["source_runs"]))
        if acceptance:
            output.parent.mkdir(parents=True, exist_ok=True)
            review_dir.mkdir(parents=True, exist_ok=False)
            with output.open("x") as stream:
                for item in selected:
                    stream.write(json.dumps({**item["pair"], "merge_source": compact(item),
                        "independent_verification": item["independent_verification"]}, sort_keys=True, allow_nan=False) + "\n")
            for kind in KINDS:
                rows = [r for r in selected if r["kind"] == kind][:REVIEWS_PER_CLASS]
                for index, row in enumerate(rows):
                    source = reference_path(row["review_image"], within=ROOT / "data/world_pairs/v2")
                    destination = review_dir / f"{kind}_{index:02d}_{row['pair_id']}.png"
                    with source.open("rb") as incoming, destination.open("xb") as outgoing:
                        shutil.copyfileobj(incoming, outgoing)
                    if sha_file(destination) != row["review_image"]["sha256"]:
                        raise ValueError("Review copy hash mismatch")
                    metrics["review_files"].append({"file": str(destination.relative_to(ROOT)), "sha256": sha_file(destination),
                                                   "pair_id": row["pair_id"], "kind": kind})
            metrics["dataset"] = {"file": str(output.relative_to(ROOT)), "sha256": sha_file(output)}
            metrics["acceptance_passed"] = len(selected) >= 80 and len(metrics["review_files"]) == 40
            metrics["status"] = "pilot_ready_for_review" if metrics["acceptance_passed"] else "output_incomplete"
            code = 0 if metrics["acceptance_passed"] else 1
        else:
            metrics["status"] = "insufficient_or_invalid_evidence"
    except Exception as error:
        metrics["source_errors"].append({"stage": "merge", "error": f"{type(error).__name__}: {error}"})
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open("x") as stream:
        json.dump({"metrics": metrics, "config": config, "sources": sources}, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print("FINAL_METRICS_JSON: " + json.dumps(metrics, sort_keys=True, allow_nan=False), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())

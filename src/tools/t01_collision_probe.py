"""Diagnostic only: compare two pinned native collision paths; never accept pairs.

Invoke via run_logged after a separate, same-PBS four-GPU 224px segmentation
render gate. Four private workers own FloorPlan203/204/205/206 respectively.
Raw metadata/actions/images stay on scratch. No existing T01 predicate changes.
"""
import argparse
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import re
import signal
import socket
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
SCENES = ("FloorPlan203", "FloorPlan204", "FloorPlan205", "FloorPlan206")
BUILD = "f0825767cd50d69f666c7f282e54abfe58f1e917"
MAX_RAW_BYTES_PER_WORKER = 256 * 1024 * 1024
# Exclude only explicit request/response bookkeeping at the metadata root.
# currentTime (Unity Time.time), unknown fields and every object field remain.
BOOKKEEPING = frozenset(("lastAction", "lastActionSuccess", "errorMessage", "errorCode",
                        "actionReturn", "sequenceId", "actionIntReturn", "actionFloatReturn",
                        "actionFloatsReturn", "actionStringsReturn", "actionVector3sReturn"))
REQUIRED_STATE = ("objects", "agent", "inventoryObjects", "isSceneAtRest")
LIMITATIONS = [
    "Diagnostic observations only: no valid_physical or accepted-pair decision is produced, and no old result is promoted.",
    "Old query returns collider names, not unique object IDs; names are never used as an ignore list.",
    "The positions[] TeleportObject overload invokes firstColliderObjectCollidingWith with forceAction=False; source triggers are excluded there.",
    "Both native paths use primitive Capsule/Box/Sphere overlaps; source MeshCollider coverage and collider.enabled state are not exposed by these observations.",
    "Overlap can report touching support geometry; no penetration depth, contact manifold, or general mesh nonpenetration proof is available.",
    "Identity teleport is an action with possible side effects, not a pure query. All non-bookkeeping metadata and exact float representations remain in comparisons.",
    "currentTime is retained and compared. A clock-only difference does not claim exact full state or absence of side effects; it permits the next diagnostic object only.",
    "CPU fixtures validate bookkeeping and failure preservation only; they cannot establish Unity API compatibility.",
]


def encoded(value):
    return json.dumps(value, sort_keys=True, allow_nan=False).encode()


def sha_file(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def strict_equal(a, b):
    return encoded(a) == encoded(b)


def state_projection(metadata):
    if not isinstance(metadata, dict) or any(key not in metadata for key in REQUIRED_STATE):
        raise ValueError("Missing required physical/semantic metadata fields")
    result = {key: copy.deepcopy(value) for key, value in metadata.items() if key not in BOOKKEEPING}
    objects = result["objects"]
    if (not isinstance(objects, list) or not all(isinstance(row, dict) for row in objects)
            or not isinstance(result["agent"], dict) or not isinstance(result["inventoryObjects"], list)
            or type(result["isSceneAtRest"]) is not bool):
        raise ValueError("Unknown physical/semantic metadata schema")
    for field in ("name", "objectId"):
        values = [row.get(field) for row in objects]
        if any(not isinstance(value, str) or not value for value in values) or len(set(values)) != len(values):
            raise ValueError("Missing or duplicate object " + field)
    # Only object-list order is normalized; all object fields and values remain.
    result["objects"] = sorted(objects, key=lambda row: row["name"])
    return result


def different_paths(a, b, path=""):
    changed, schema = [], []
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            child = path + "/" + key.replace("~", "~0").replace("/", "~1")
            if key not in a or key not in b:
                changed.append(child)
                schema.append(child)
            else:
                values, missing = different_paths(a[key], b[key], child)
                changed.extend(values)
                schema.extend(missing)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            changed.append(path)
            schema.append(path)
        else:
            for index, (left, right) in enumerate(zip(a, b)):
                values, missing = different_paths(left, right, path + "/" + str(index))
                changed.extend(values)
                schema.extend(missing)
    elif not strict_equal(a, b):
        changed.append(path)
        if type(a) is not type(b):
            schema.append(path)
    return changed, schema


def compare_state(before, after):
    try:
        left, right = state_projection(before), state_projection(after)
        changed, schema = different_paths(left, right)
        core = {key: strict_equal(left[key], right[key]) for key in REQUIRED_STATE}
        non_clock = [path for path in changed if path != "/currentTime"]
        return {"comparison_available": True, "full_state_exact": not changed,
                "required_state_fields_exact": core, "changed_paths": changed,
                "schema_change_paths": schema, "metadata_side_effect_observed": bool(changed),
                "clock_only_difference": bool(changed) and not non_clock and not schema,
                "safe_to_continue_diagnostic": not non_clock and not schema,
                "unknown_geometry_coverage": True,
                "excluded_request_bookkeeping": sorted(BOOKKEEPING), "unavailable_reason": None}
    except (TypeError, ValueError, KeyError) as error:
        return {"comparison_available": False, "full_state_exact": None,
                "required_state_fields_exact": None, "changed_paths": None,
                "schema_change_paths": None, "metadata_side_effect_observed": None,
                "clock_only_difference": None, "safe_to_continue_diagnostic": False,
                "unknown_geometry_coverage": True, "excluded_request_bookkeeping": sorted(BOOKKEEPING),
                "unavailable_reason": f"{type(error).__name__}: {error}"}


def teleport_arguments(obj):
    # An array is essential: the position overload starts a settle coroutine.
    return {"objectId": obj["objectId"], "positions": [copy.deepcopy(obj["position"])],
            "rotation": copy.deepcopy(obj["rotation"]), "forceAction": False,
            "forceKinematic": False, "allowTeleportOutOfHand": False, "makeUnbreakable": False}


class EvidenceStore:
    def __init__(self, directory, max_bytes=MAX_RAW_BYTES_PER_WORKER):
        self.directory, self.max_bytes = Path(directory), max_bytes
        self.directory.mkdir(parents=True, exist_ok=False)
        self.files, self.bytes = [], 0
        # Each completed file is indexed immediately, so a deadline/native crash
        # retains a small verifiable manifest even before the final index exists.
        self.incremental = self.directory / "evidence_index.jsonl"
        with self.incremental.open("xb"):
            pass

    def save(self, name, content):
        path = self.directory / name
        if path.parent != self.directory or self.bytes + len(content) > self.max_bytes:
            raise ValueError("Raw evidence path or per-worker byte limit exceeded; prior files retained")
        with path.open("xb") as stream:
            stream.write(content)
        reference = {"file": str(path), "sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}
        self.files.append(reference)
        self.bytes += len(content)
        with self.incremental.open("ab") as index:
            index.write(encoded(reference) + b"\n")
        return reference

    def save_json(self, name, value):
        return self.save(name, encoded(value) + b"\n")

    def index(self):
        return self.save_json("evidence_index.json", {"files": list(self.files), "raw_bytes": self.bytes,
                             "max_raw_bytes": self.max_bytes, "raw_files_not_deleted": True})


class BlackFrame(RuntimeError):
    pass


class ActionRecorder:
    """Persist each requested action and both full metadata snapshots, even failures."""
    def __init__(self, controller, store, frame_capture, deadline):
        self.controller, self.store = controller, store
        self.frame_capture, self.deadline = frame_capture, deadline
        self.last_record, self.action_count = None, 0

    def execute(self, action, arguments=None, require_success=True):
        if time.time() >= self.deadline:
            raise TimeoutError("Collision probe worker deadline reached")
        arguments = copy.deepcopy(arguments or {})
        before = copy.deepcopy(self.controller.last_event.metadata)
        before_rgb = self.frame_capture(self.controller.last_event, f"before_{action}")
        row = {"index": self.action_count, "action": action, "arguments": arguments,
               "before_metadata": before, "before_rgb": before_rgb, "after_metadata": None,
               "after_rgb": None, "exception": None, "lastActionSuccess": None}
        self.action_count += 1
        event = None
        try:
            if before_rgb.get("black"):
                raise BlackFrame("Black pre-action RGB; no further action sent")
            if before_rgb.get("valid") is not True:
                raise RuntimeError("Invalid pre-action RGB; no further action sent")
            event = self.controller.step(action=action, **arguments)
            row.update(after_metadata=copy.deepcopy(event.metadata), after_rgb=self.frame_capture(event, f"after_{action}"),
                       lastActionSuccess=event.metadata.get("lastActionSuccess"),
                       errorMessage=event.metadata.get("errorMessage"), actionReturn=copy.deepcopy(event.metadata.get("actionReturn")))
        except BaseException as error:
            row["exception"] = f"{type(error).__name__}: {error}"
            if row["after_metadata"] is None:
                row["last_available_metadata_after_exception"] = copy.deepcopy(self.controller.last_event.metadata)
            raise
        finally:
            if row["after_metadata"] is not None:
                row["state_comparison"] = compare_state(before, row["after_metadata"])
                row["rgb_bytes_exact"] = (before_rgb.get("rgb_sha256") == (row["after_rgb"] or {}).get("rgb_sha256"))
            self.last_record = {"reference": self.store.save_json(f"action_{row['index']:05d}.json", row), "record": row}
        if any(value and value.get("black") for value in (before_rgb, row["after_rgb"])):
            raise BlackFrame("Measured all-zero RGB event frame; diagnostic stopped")
        if any(not value or value.get("valid") is not True for value in (before_rgb, row["after_rgb"])):
            raise RuntimeError("Missing or invalid event RGB; no physical conclusion")
        if require_success and row["lastActionSuccess"] is not True:
            raise RuntimeError(f"{action}: {row.get('errorMessage')}")
        return event


def probe_object(recorder, obj):
    before = recorder.controller.last_event.metadata
    if before.get("inventoryObjects") != []:
        raise RuntimeError("Native identity teleport probe requires explicitly empty hands")
    result = {"object_name": obj["name"], "object_id": obj["objectId"],
              "old_query": None, "identity_teleport": None, "complete": False,
              "safe_to_continue_diagnostic": False, "physical_validity": None}
    for label, action, arguments in (
        ("old_query", "CollidersObjectCollidingWith", {"objectId": obj["objectId"]}),
        ("identity_teleport", "TeleportObject", teleport_arguments(obj)),
    ):
        event = recorder.execute(action, arguments, require_success=False)
        recorded = recorder.last_record
        metadata = event.metadata
        comparison = recorded["record"]["state_comparison"]
        raw = metadata.get("actionReturn")
        schema_ok = isinstance(raw, list) and all(isinstance(value, str) for value in raw) if label == "old_query" else None
        result[label] = {"action_record": recorded["reference"],
                         "lastActionSuccess": metadata.get("lastActionSuccess"),
                         "errorMessage": metadata.get("errorMessage"), "actionReturn": copy.deepcopy(raw),
                         "known_return_schema": schema_ok, "state_comparison": comparison,
                         "rgb_bytes_exact": recorded["record"]["rgb_bytes_exact"]}
        success = metadata.get("lastActionSuccess")
        collision_rejected = (label == "identity_teleport" and success is False
            and isinstance(metadata.get("errorMessage"), str)
            and metadata["errorMessage"].startswith(obj["objectId"] + " is colliding with ")
            and metadata["errorMessage"].endswith(" after teleport."))
        result[label]["native_collision_rejected"] = collision_rejected
        api_response_known = (metadata.get("lastAction") == action and type(success) is bool
                              and (success or collision_rejected))
        result[label]["api_response_known"] = api_response_known
        if (not comparison["safe_to_continue_diagnostic"] or not api_response_known or schema_ok is False):
            result["stopped_reason"] = "old API failure/schema unknown or reported state changed beyond clock"
            return result
    # A false native teleport success flag is a measured diagnostic outcome,
    # not an execution exception or permission to ignore a collider.
    result["complete"] = True
    result["safe_to_continue_diagnostic"] = result["identity_teleport"]["state_comparison"]["safe_to_continue_diagnostic"]
    return result


def compact_case(row):
    result = {key: row.get(key) for key in ("object_name", "object_id", "complete", "safe_to_continue_diagnostic", "stopped_reason")}
    for key in ("old_query", "identity_teleport"):
        source = row.get(key)
        result[key] = None if source is None else {
            "action_record": source["action_record"], "lastActionSuccess": source["lastActionSuccess"],
            "errorMessage": source["errorMessage"], "known_return_schema": source["known_return_schema"],
            "api_response_known": source["api_response_known"], "native_collision_rejected": source["native_collision_rejected"],
            "full_state_exact": source["state_comparison"]["full_state_exact"],
            "required_state_fields_exact": source["state_comparison"]["required_state_fields_exact"],
            "metadata_side_effect_observed": source["state_comparison"]["metadata_side_effect_observed"],
            "clock_only_difference": source["state_comparison"]["clock_only_difference"],
            "rgb_bytes_exact": source["rgb_bytes_exact"],
            "old_collider_names": source["actionReturn"] if key == "old_query" else None}
    return result


def probe_worker(config, connection):
    """One private Unity worker on one already verified allocation logical slot."""
    from src.envs.thor_allocation import make_controller_class
    from src.envs.thor_common import process_identity
    from src.envs.pair_simulator_approved import PairSimulator, SETTLING
    from ai2thor.platform import CloudRendering
    from PIL import Image
    import numpy as np

    runtime, logical = copy.deepcopy(config["runtime"]), config["logical_index"]
    scene, directory = config["scene"], Path(config["raw_directory"])
    metrics = {"scene": scene, "logical_index": logical, "gpu_identity": runtime["devices"][logical],
               "pbs_jobid": runtime["pbs_jobid"], "node": socket.gethostname(), "status": "initializing",
               "diagnostic_completed": False, "black_frame_detected": False, "black_frame_stop": None,
               "settled": None, "objects_expected": None, "objects": [], "errors": [], "cleanup_errors": [],
               "raw_directory": str(directory), "evidence_index": None, "physical_validity": None}
    store, controller, recorder = None, None, None
    images = {}

    def publish():
        connection.send(("progress", copy.deepcopy(metrics)))

    def capture(event, stage):
        frame = event.frame
        valid = frame is not None and frame.shape == (224, 224, 3) and frame.dtype == np.uint8
        observation = {"shape": list(frame.shape) if frame is not None else None,
                       "dtype": str(frame.dtype) if frame is not None else None,
                       "nonzero_values": int(np.count_nonzero(frame)) if valid else None,
                       "min": int(frame.min()) if valid else None, "max": int(frame.max()) if valid else None,
                       "black": bool(valid and not np.any(frame)), "valid": bool(valid and np.any(frame)),
                       "scope": "returned event RGB; fresh FIFO rendering count is not inferred"}
        if valid:
            identity = hashlib.sha256(frame.tobytes()).hexdigest()
            if identity not in images:
                buffer = io.BytesIO()
                Image.fromarray(frame).save(buffer, format="PNG")
                images[identity] = store.save("rgb_" + identity + ".png", buffer.getvalue())
            observation.update(rgb_sha256=identity, image=images[identity])
        if observation["black"] and metrics["black_frame_stop"] is None:
            raw = {"measurement": observation, "stage": stage, "scene": scene,
                   "pbs_jobid": runtime["pbs_jobid"], "gpu_identity": runtime["devices"][logical],
                   "logical_index": logical, "node": socket.gethostname(),
                   "image": observation["image"]["file"], "image_sha256": observation["image"]["sha256"],
                   "metadata": copy.deepcopy(event.metadata)}
            reference = store.save_json("black_frame.json", raw)
            metrics.update(black_frame_detected=True, black_frame_stop={**reference, **{key: raw[key] for key in
                ("measurement", "stage", "scene", "pbs_jobid", "gpu_identity", "logical_index", "node", "image", "image_sha256")}})
            publish()
        return observation

    try:
        if scene != SCENES[logical] or os.environ.get("PBS_JOBID") != runtime["pbs_jobid"] or os.environ.get("CUDA_VISIBLE_DEVICES") != runtime["cuda_visible_devices"]:
            raise ValueError("Worker scene or original PBS allocation identity mismatch")
        cpus = set(config["cpu_affinity"])
        if len(cpus) != 12 or not cpus <= os.sched_getaffinity(0):
            raise ValueError("Require twelve assigned affinity CPUs per worker")
        os.sched_setaffinity(0, cpus)
        store = EvidenceStore(directory)
        metrics["incremental_index_path"] = str(store.incremental)
        store.save_json("worker_config.json", config)
        publish()
        runtime["log_dir"] = str(directory / "unity_logs")
        cls = make_controller_class(runtime, role="collision_probe")
        controller = cls.__new__(cls)
        controller.__init__(scene=scene, platform=CloudRendering, gpu_device=logical,
            width=224, height=224, agentMode="default", renderDepthImage=False,
            renderInstanceSegmentation=True, gridSize=0.1, visibilityDistance=10,
            fieldOfView=100, server_timeout=30, server_start_timeout=60)
        connection.send(("unity_identity", process_identity(controller.unity_pid)))
        if getattr(getattr(controller, "_build", None), "commit_id", None) != BUILD:
            raise RuntimeError("Unexpected Unity build; no API conclusion")
        initial_rgb = capture(controller.last_event, "constructor_return")
        store.save_json("constructor.json", {"metadata": copy.deepcopy(controller.last_event.metadata), "rgb": initial_rgb})
        if initial_rgb["black"]:
            raise BlackFrame("Black constructor return RGB")
        if not initial_rgb["valid"]:
            raise RuntimeError("Invalid constructor return RGB")
        recorder = ActionRecorder(controller, store, capture, config["deadline"])

        class LoggedSimulator(PairSimulator):
            def action(self, action, **kwargs):
                return recorder.execute(action, kwargs)

        sim = LoggedSimulator(controller, {"scene": scene, "deadline": config["deadline"]})
        sim.action("InitialRandomSpawn", randomSeed=0, forceVisible=False, placeStationary=True, numPlacementAttempts=5)
        source = sim.world_from_event(scene + "_source")
        store.save_json("source_world.json", source)
        world, settling = sim.settle_world(source, scene + "_settled")
        metrics["settled"] = settling["settled"]
        metrics["settling_evidence"] = store.save_json("settling.json", {"world": world, "settling": settling, "unchanged_policy": SETTLING})
        if not settling["settled"]:
            raise RuntimeError("Original approved settling did not pass; native comparison not attempted")
        sim.action("PausePhysicsAutoSim")
        sim.action("PhysicsSyncTransforms")
        if controller.last_event.metadata.get("inventoryObjects") != []:
            raise RuntimeError("Explicitly empty inventory required")
        baseline = copy.deepcopy(controller.last_event.metadata)
        store.save_json("comparison_baseline.json", baseline)
        state_projection(baseline)
        objects = sorted((obj for obj in baseline["objects"] if obj.get("moveable") or obj.get("pickupable")), key=lambda row: row["name"])
        metrics.update(status="comparing", objects_expected=len(objects))
        publish()
        for index, obj in enumerate(objects):
            sim.action("PausePhysicsAutoSim")
            sim.action("PhysicsSyncTransforms")
            # Read the current identity/pose, never reuse a stale objectId after a mutation.
            current = [row for row in controller.last_event.metadata["objects"] if row["name"] == obj["name"]]
            if len(current) != 1:
                raise RuntimeError("Object identity changed during diagnostic")
            case = probe_object(recorder, current[0])
            case_ref = store.save_json(f"object_{index:04d}.json", case)
            metrics["objects"].append({**compact_case(case), "raw_case": case_ref})
            publish()
            if not case["complete"] or not case["safe_to_continue_diagnostic"]:
                raise RuntimeError("Native comparison incomplete or state changed; further objects not tested")
        metrics.update(diagnostic_completed=True, status="completed_diagnostic")
    except BlackFrame as error:
        metrics.update(status="black_frame_stop", diagnostic_completed=False)
        metrics["errors"].append(f"BlackFrame: {error}")
    except BaseException as error:
        metrics.update(status="incomplete_diagnostic", diagnostic_completed=False)
        metrics["errors"].append(f"{type(error).__name__}: {error}")
        traceback.print_exc()
    finally:
        publish()
        if controller is not None:
            signal.setitimer(signal.ITIMER_REAL, 5)
            try:
                controller.stop()
            except BaseException as error:
                metrics["cleanup_errors"].append(f"{type(error).__name__}: {error}")
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
        if store is not None:
            try:
                metrics["recorded_actions"] = recorder.action_count if recorder is not None else None
                metrics["evidence_index"] = store.index()
            except BaseException as error:
                metrics["errors"].append(f"Evidence index: {type(error).__name__}: {error}")
                metrics["diagnostic_completed"] = False
        publish()
    return metrics


def validate_gate(data, environment):
    metrics, config, meta = data["metrics"], data["config"], data["_meta"]
    runtime = config.get("allocation") or {}
    devices = runtime.get("devices", [])
    if (meta.get("exit_code") != 0 or metrics.get("passed") is not True or metrics.get("errors")
            or metrics.get("cleanup_errors") or metrics.get("expected_gpu_count") != 4
            or metrics.get("completed_gpu_count") != 4 or config.get("resolution") != 224
            or config.get("segmentation") is not True or config.get("pbs_jobid") != environment.get("PBS_JOBID")
            or runtime.get("pbs_jobid") != environment.get("PBS_JOBID")
            or runtime.get("cuda_visible_devices") != environment.get("CUDA_VISIBLE_DEVICES")
            or runtime.get("scratch") != str(Path(environment.get("CVPR_SCRATCH", "")).resolve())
            or [row.get("logical_index") for row in devices] != [0, 1, 2, 3]
            or len({row.get("gpu_uuid") for row in devices}) != 4):
        raise ValueError("Require an audited successful same-PBS four-GPU 224px segmentation gate")
    return copy.deepcopy(runtime)


def preserve_black_evidence(row, directory):
    stop = row.get("black_frame_stop")
    if not stop:
        return
    directory.mkdir(parents=True, exist_ok=False)
    for path_key, hash_key, target_key in (("file", "sha256", "portable_file"), ("image", "image_sha256", "portable_image")):
        source = Path(stop[path_key])
        if source.stat().st_size > 32 * 1024 * 1024:
            raise ValueError("Portable black evidence exceeds the small-artifact limit; raw scratch file retained")
        if sha_file(source) != stop[hash_key]:
            raise ValueError("Black-frame raw evidence hash mismatch")
        destination = directory / source.name
        with source.open("rb") as original, destination.open("xb") as output:
            for block in iter(lambda: original.read(1024 * 1024), b""):
                output.write(block)
        copied_sha = sha_file(destination)
        if copied_sha != stop[hash_key]:
            raise ValueError("Portable black-frame evidence hash mismatch; raw and copied files retained")
        stop[target_key] = str(destination.relative_to(ROOT))
        stop[target_key + "_sha256"] = copied_sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--gate-result", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=180)
    args = parser.parse_args()
    if not re.fullmatch(r"t01_[A-Za-z0-9_]+", args.name) or not 30 <= args.timeout_seconds <= 300:
        parser.error("Use a unique t01_ name and a 30..300 second total work/cleanup window")
    output = ROOT / "results" / f"{args.name}.json"
    portable = ROOT / "logs/runs" / args.name
    if output.exists() or portable.exists():
        raise FileExistsError("Preserve earlier diagnostic evidence; choose a unique name")
    from src.envs.thor_common import Worker, install_signals, stop_workers
    from src.tools.t01_partition import audited_result, cpu_shards
    metrics = {"kind": "t01_collision_probe", "diagnostic_completed": False, "black_frame_detected": False,
               "workers": [], "errors": [], "cleanup_errors": [], "physical_validity": None,
               "task_acceptance": None, "exit_code": 1}
    config = {"name": args.name, "pbs_jobid": os.environ.get("PBS_JOBID"), "scenes": list(SCENES),
              "timeout_seconds": args.timeout_seconds, "limitations": LIMITATIONS,
              "state_bookkeeping_exclusions": sorted(BOOKKEEPING), "sources": [], "gate_source": None,
              "pinned_unity_build": BUILD,
              "native_sources": {"teleport_positions_overload": "https://github.com/allenai/ai2thor/blob/" + BUILD + "/unity/Assets/Scripts/PhysicsRemoteFPSAgentController.cs#L661",
                  "collider_helpers": "https://github.com/allenai/ai2thor/blob/" + BUILD + "/unity/Assets/Scripts/UtilityFunctions.cs#L86"}}
    workers, worker_configs, raw_root = [], [], None
    work_deadline = time.time() + args.timeout_seconds - 10
    install_signals()
    signal.setitimer(signal.ITIMER_REAL, args.timeout_seconds)
    try:
        data, config["gate_source"] = audited_result(args.gate_result)
        runtime = validate_gate(data, os.environ)
        config["allocation"] = runtime
        affinity = cpu_shards(os.sched_getaffinity(0), 4)
        raw_root = Path(runtime["scratch"]) / "t01_collision_probe" / args.name / runtime["pbs_jobid"]
        raw_root.mkdir(parents=True, exist_ok=False)
        config["raw_directory"] = str(raw_root)
        for source in (__file__, ROOT / "src/envs/pair_simulator_approved.py", ROOT / "src/envs/pair_simulator.py",
                       ROOT / "src/envs/pair_approved_policy.py", ROOT / "src/envs/thor_allocation.py"):
            config["sources"].append({"file": str(Path(source).relative_to(ROOT)), "sha256": sha_file(source)})
        deadline = work_deadline
        for logical, scene in enumerate(SCENES):
            child_config = {"runtime": runtime, "logical_index": logical,
                "scene": scene, "cpu_affinity": affinity[logical], "deadline": deadline,
                "raw_directory": str(raw_root / scene)}
            worker_configs.append(child_config)
            workers.append(Worker(probe_worker, child_config))
        while True:
            for worker in workers:
                worker.receive()
            if any(worker.latest.get("black_frame_detected") for worker in workers):
                metrics["black_frame_detected"] = True
                metrics["black_frame_stop_all_workers"] = True
                break
            if all(worker.result is not None or worker.error is not None for worker in workers):
                break
            if time.time() >= deadline:
                raise TimeoutError("Collision probe parent work deadline reached; cleanup reserve follows")
            time.sleep(0.05)
    except BaseException as error:
        metrics["errors"].append(f"{type(error).__name__}: {error}")
        traceback.print_exc()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        metrics["cleanup_errors"] = stop_workers(workers)
        for worker, child_config in zip(workers, worker_configs):
            row = copy.deepcopy(worker.latest)
            row["assigned_config"] = {key: child_config[key] for key in ("scene", "logical_index", "cpu_affinity", "raw_directory")}
            row["worker_error"] = worker.error
            row["worker_returned_result"] = worker.result is not None
            incremental = Path(child_config["raw_directory"]) / "evidence_index.jsonl"
            if incremental.exists() and not metrics["cleanup_errors"]:
                row["incremental_evidence_index"] = {"file": str(incremental), "sha256": sha_file(incremental),
                    "bytes": incremental.stat().st_size, "complete_worker_result": worker.result is not None}
            if row.get("black_frame_stop"):
                try:
                    preserve_black_evidence(row, portable / f"gpu_{row['logical_index']}" / "black_frame")
                except Exception as error:
                    metrics["errors"].append(f"Black evidence preservation: {type(error).__name__}: {error}")
            metrics["workers"].append(row)
        metrics["black_frame_detected"] |= any(row.get("black_frame_detected") for row in metrics["workers"])
        cleanup_bad = metrics["cleanup_errors"] or any(row.get("cleanup_errors") for row in metrics["workers"])
        complete = len(workers) == 4 and all(row.get("diagnostic_completed") and row.get("worker_returned_result")
            and not row.get("worker_error") and not row.get("errors") for row in metrics["workers"])
        metrics["diagnostic_completed"] = bool(complete and not metrics["errors"] and not cleanup_bad)
        metrics["exit_code"] = (86 if metrics["black_frame_detected"] and not cleanup_bad and not metrics["errors"]
                                else 0 if metrics["diagnostic_completed"] else 1)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x") as stream:
            json.dump({"metrics": metrics, "config": config}, stream, indent=2, allow_nan=False)
            stream.write("\n")
        print("FINAL_METRICS_JSON: " + json.dumps(metrics, sort_keys=True, allow_nan=False), flush=True)
    return metrics["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())

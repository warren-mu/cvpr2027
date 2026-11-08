"""Evidence-only black-frame retry guard for the fixed collision/packing wrapper.

No submission, process execution, accounting, or result mutation happens here.
All child results must pass phase_watch.load_logged before their data is used.
Ordinary diagnostic/API failures are never a reason to retry automatically.
"""
import hashlib
import io
import json
from pathlib import Path
import re

SCENES = ("FloorPlan203", "FloorPlan204", "FloorPlan205", "FloorPlan206")
PYTHONS = ("python", "python3", "python3.10")
MAX_EVIDENCE_BYTES = 32 * 1024 * 1024


def python_command(command, arguments):
    return (isinstance(command, list) and len(command) == len(arguments) + 1
            and Path(command[0]).name in PYTHONS and command[1:] == arguments)


def require_outer(repo, payload, job_id, fields, outer):
    name, command = payload["name"], payload["command"]
    inner = ["bash", "experiments/t01_collision_pack.sh", "--name", name]
    expected = ["src/tools/record_command.py", "--name", name, "--", *inner]
    if (not re.fullmatch(r"t01_[A-Za-z0-9_]+", name) or not python_command(command, expected)
            or payload.get("requested_gpu_count") != 4
            or fields.get("project") != "CFP04-SF-125" or fields.get("job_state") != "F"
            or fields.get("Exit_status") != "86"
            or outer.get("_meta", {}).get("name") != name or outer["_meta"].get("exit_code") != 86
            or outer["_meta"].get("command") != command
            or outer.get("config", {}).get("pbs_job_id") != job_id
            or outer["config"].get("command") != inner
            or outer.get("metrics", {}).get("command_exit_code") != 86 or outer.get("error") is not None):
        raise ValueError("Require the exact four-GPU collision/pack outer command and same-job terminal exit 86")


def gate_header(payload, job_id, fields, gate, name, resolution, segmentation):
    """Check an actual gate's command and allocation; never fabricate a gate."""
    meta, config = gate.get("_meta", {}), gate.get("config", {})
    arguments = ["-m", "src.tools.render_gate", "--name", name, "--gpus", "4",
                 "--resolution", str(resolution)]
    if segmentation:
        arguments.append("--segmentation")
    arguments += ["--timeout-seconds", "180"]
    allocation = config.get("allocation") or {}
    devices = allocation.get("devices", [])
    nodes = {part.split("/")[0] for part in fields.get("exec_host", "").split("+") if part}
    scratch = str(Path(payload["environment"]["CVPR_SCRATCH"]).resolve())
    if (meta.get("name") != name or not python_command(meta.get("command"), arguments)
            or config.get("name") != name or config.get("pbs_jobid") != job_id
            or config.get("resolution") != resolution or config.get("segmentation") is not segmentation
            or config.get("timeout_seconds") != 180 or nodes != {config.get("node")}
            or allocation.get("pbs_jobid") != job_id or allocation.get("scratch") != scratch
            or not isinstance(allocation.get("cuda_visible_devices"), str)
            or not allocation["cuda_visible_devices"]
            or len(devices) != 4 or [row.get("logical_index") for row in devices] != list(range(4))
            or any(not isinstance(row.get("gpu_uuid"), str) or not row["gpu_uuid"] for row in devices)
            or len({row["gpu_uuid"] for row in devices}) != 4):
        raise ValueError("Gate command, node, or four-GPU allocation differs from the fixed wrapper")
    return allocation


def successful_gate(payload, job_id, fields, gate, name):
    allocation = gate_header(payload, job_id, fields, gate, name, 224, True)
    metrics = gate.get("metrics", {})
    observed = metrics.get("devices", [])
    if (gate["_meta"].get("exit_code") != 0 or metrics.get("exit_code") != 0
            or metrics.get("passed") is not True or metrics.get("black_frame_detected") is not False
            or metrics.get("errors") != [] or metrics.get("cleanup_errors") != []
            or metrics.get("expected_gpu_count") != 4 or metrics.get("completed_gpu_count") != 4
            or len(observed) != 4):
        raise ValueError("Initial successful gate is missing complete clean evidence")
    for identity, row in zip(allocation["devices"], observed):
        observation = row.get("observation", {})
        captures = observation.get("captures", [])
        if (row.get("worker_error") is not None or observation.get("error") is not None
                or observation.get("cleanup_error") is not None or observation.get("valid") is not True
                or observation.get("node") != gate["config"]["node"]
                or observation.get("gpu_identity") != identity or len(captures) != 2):
            raise ValueError("Initial gate per-GPU identity/cleanup is incomplete")
        for capture in captures:
            if (capture.get("valid_shape") is not True or capture.get("valid") is not True
                    or capture.get("black") is not False or capture.get("action_success") is not True
                    or capture.get("shape") != [224, 224, 3] or capture.get("dtype") != "uint8"
                    or type(capture.get("nonzero_rgb_values")) is not int or capture["nonzero_rgb_values"] <= 0):
                raise ValueError("Initial gate has no valid nonblack capture on every GPU")
    return allocation


def same_allocation(left, right):
    return all(left.get(key) == right.get(key) for key in
               ("pbs_jobid", "scratch", "cuda_visible_devices", "devices", "cpu_affinity"))


def source_matches(repo, source, reference):
    path = source.get("path", source.get("file"))
    return (isinstance(path, str) and (repo / path).resolve() == (repo / reference["path"]).resolve()
            and source.get("sha256") == reference["sha256"])


def validate_probe(repo, payload, job_id, gate, gate_reference, data):
    name = payload["name"] + "_probe"
    config, metrics, meta = data.get("config", {}), data.get("metrics", {}), data.get("_meta", {})
    code = meta.get("exit_code")
    arguments = ["-m", "src.tools.t01_collision_probe", "--name", name,
                 "--gate-result", f"results/{payload['name']}_gate.json", "--timeout-seconds", "180"]
    allocation = gate["config"]["allocation"]
    raw_root = Path(allocation["scratch"]) / "t01_collision_probe" / name / job_id
    workers = metrics.get("workers")
    if (type(code) is not int or code not in (0, 1, 86) or metrics.get("exit_code") != code
            or meta.get("name") != name or not python_command(meta.get("command"), arguments)
            or config.get("name") != name or config.get("pbs_jobid") != job_id
            or config.get("scenes") != list(SCENES) or config.get("timeout_seconds") != 180
            or config.get("raw_directory") != str(raw_root)
            or not source_matches(repo, config.get("gate_source", {}), gate_reference)
            or not same_allocation(allocation, config.get("allocation") or {})
            or metrics.get("kind") != "t01_collision_probe" or metrics.get("cleanup_errors") != []
            or type(metrics.get("black_frame_detected")) is not bool
            or not isinstance(workers, list) or len(workers) != 4):
        raise ValueError("Probe child command/provenance/allocation/cleanup is incomplete")
    seen_cpus = set()
    for index, (row, identity, scene) in enumerate(zip(workers, allocation["devices"], SCENES)):
        assigned = row.get("assigned_config", {})
        cpus = assigned.get("cpu_affinity", [])
        raw_directory = str(raw_root / scene)
        if (row.get("logical_index") != index or row.get("scene") != scene
                or row.get("pbs_jobid") != job_id or row.get("node") != gate["config"]["node"]
                or row.get("gpu_identity") != identity or row.get("raw_directory") != raw_directory
                or row.get("cleanup_errors") != [] or row.get("worker_error") is not None
                or type(row.get("black_frame_detected")) is not bool
                or assigned.get("logical_index") != index or assigned.get("scene") != scene
                or assigned.get("raw_directory") != raw_directory or not isinstance(cpus, list)
                or len(cpus) != 12 or any(type(cpu) is not int for cpu in cpus)
                or len(set(cpus)) != 12 or seen_cpus.intersection(cpus)
                or not set(cpus).issubset(set(allocation.get("cpu_affinity", [])))):
            raise ValueError("Probe does not retain four fixed, disjoint worker assignments with reliable cleanup")
        seen_cpus.update(cpus)
    black = any(row["black_frame_detected"] for row in workers)
    if metrics["black_frame_detected"] is not black:
        raise ValueError("Probe aggregate black status disagrees with worker observations")
    if code == 86:
        if not black or metrics.get("errors") != []:
            raise ValueError("Probe exit 86 has missing black evidence or an unrelated parent failure")
    elif black or any(row.get("black_frame_stop") is not None or row.get("worker_returned_result") is not True for row in workers):
        raise ValueError("Pack continuation requires nonblack completed workers, not uncertain cleanup")
    return workers


def confined(path, base):
    result = Path(path).resolve()
    result.relative_to(base.resolve())
    if result == base.resolve():
        raise ValueError("Evidence must be a file inside the assigned directory")
    return result


def read_small(path):
    if path.stat().st_size > MAX_EVIDENCE_BYTES:
        raise ValueError("Black evidence exceeds the bounded artifact size")
    return path.read_bytes()


def black_worker_evidence(repo, row, probe_name):
    """Verify raw/portable bytes, their logged identity, and decoded RGB pixels."""
    from PIL import Image
    stop = row.get("black_frame_stop") or {}
    raw_root = Path(row["raw_directory"])
    raw_original = confined(stop["file"], raw_root)
    image_original = confined(stop["image"], raw_root)
    portable = bool(stop.get("portable_file") or stop.get("portable_image"))
    if portable:
        directory = repo / "logs/runs" / probe_name / f"gpu_{row['logical_index']}" / "black_frame"
        raw = confined(repo / stop["portable_file"], directory)
        image = confined(repo / stop["portable_image"], directory)
        if (raw.name != raw_original.name or image.name != image_original.name
                or stop.get("portable_file_sha256") != stop.get("sha256")
                or stop.get("portable_image_sha256") != stop.get("image_sha256")):
            raise ValueError("Portable capture does not match its exact original evidence")
    else:
        raw, image = raw_original, image_original
    raw_bytes, image_bytes = read_small(raw), read_small(image)
    if (hashlib.sha256(raw_bytes).hexdigest() != stop["sha256"]
            or hashlib.sha256(image_bytes).hexdigest() != stop["image_sha256"]
            or stop.get("bytes") != len(raw_bytes)):
        raise ValueError("Black capture raw/PNG bytes differ from the logged hashes")
    record = json.loads(raw_bytes)
    fields = ("measurement", "stage", "scene", "pbs_jobid", "gpu_identity", "logical_index", "node", "image", "image_sha256")
    if (any(record.get(key) != stop.get(key) for key in fields)
            or set(record) != set(fields) | {"metadata"} or not isinstance(record["metadata"], dict)
            or any(stop.get(key) != row.get(key) for key in
                   ("scene", "pbs_jobid", "gpu_identity", "logical_index", "node"))
            or not isinstance(stop.get("stage"), str) or not stop["stage"]):
        raise ValueError("Black capture metadata/identity differs from its worker")
    measured = stop["measurement"]
    if (measured.get("shape") != [224, 224, 3] or measured.get("dtype") != "uint8"
            or measured.get("black") is not True or measured.get("valid") is not False
            or any(type(measured.get(key)) is not int or measured[key] != 0 for key in ("min", "max", "nonzero_values"))
            or measured.get("image", {}).get("file") != stop["image"]
            or measured["image"].get("sha256") != stop["image_sha256"]
            or measured["image"].get("bytes") != len(image_bytes)):
        raise ValueError("Missing exact 224px uint8 black RGB measurement")
    with Image.open(io.BytesIO(image_bytes)) as decoded:
        if decoded.format != "PNG" or decoded.mode != "RGB" or decoded.size != (224, 224):
            raise ValueError("Black evidence is not the declared 224px RGB PNG")
        decoded.load()
        if decoded.getextrema() != ((0, 0), (0, 0), (0, 0)):
            raise ValueError("PNG contains nonzero RGB despite a black-frame claim")
        if hashlib.sha256(decoded.tobytes()).hexdigest() != measured.get("rgb_sha256"):
            raise ValueError("Decoded RGB differs from the original frame hash")
    return {"worker_logical_index": row["logical_index"], "scene": row["scene"],
            "gpu_uuid": row["gpu_identity"]["gpu_uuid"], "file": str(raw), "sha256": stop["sha256"],
            "image": str(image), "image_sha256": stop["image_sha256"], "portable_copy": portable,
            "decoded_rgb_verified_all_zero": True}


def validate_probe_pack_retry(repo, payload, job_id, fields, outer, gate, gate_reference, auditor):
    from src.tools.phase_watch import load_logged, validate_black_retry
    repo = repo.resolve()
    require_outer(repo, payload, job_id, fields, outer)
    name, gate_name = payload["name"], payload["name"] + "_gate"
    gate_header(payload, job_id, fields, gate, gate_name, 224, True)
    if gate.get("_meta", {}).get("exit_code") == 86:
        reason = validate_black_retry(payload, job_id, fields, outer, gate, gate_name, "probe_pack")
        return "probe_pack_initial_" + reason, {"initial_gate": gate_reference}
    allocation = successful_gate(payload, job_id, fields, gate, gate_name)
    probe_name = name + "_probe"
    probe, probe_ref = load_logged(repo, repo / "results" / f"{probe_name}.json", auditor)
    workers = validate_probe(repo, payload, job_id, gate, gate_reference, probe)
    if probe["_meta"]["exit_code"] == 86:
        evidence = [black_worker_evidence(repo, row, probe_name) for row in workers if row["black_frame_detected"]]
        return "probe_pack_probe_black_capture", {"initial_gate": gate_reference, "probe": probe_ref,
                                                   "black_captures": evidence}
    pack_name = "t00b_" + name + "_pack"
    pack, pack_ref = load_logged(repo, repo / "results" / f"{pack_name}.json", auditor)
    pack_gate, pack_gate_ref = load_logged(repo, repo / "results" / f"{pack_name}_gate.json", auditor)
    command = ["bash", "experiments/t00b_pack_smoke.sh", "--name", pack_name,
               "--seconds", "120", "--task-timeout", "240", "--worker-timeout", "360", "--gate-timeout", "180"]
    config, metrics = pack.get("config", {}), pack.get("metrics", {})
    pack_allocation = config.get("allocation") or {}
    pack_gate_allocation = gate_header(payload, job_id, fields, pack_gate, pack_name + "_gate", 255, False)
    if (pack.get("_meta", {}).get("command") != command or config.get("name") != pack_name
            or metrics.get("kind") != "synthetic_packing_smoke_not_research" or metrics.get("gate_exit_code") != 86
            or pack_allocation.get("pbs_jobid") != job_id or pack_allocation.get("name") != pack_name
            or pack_allocation.get("host") != gate["config"]["node"]
            or pack_allocation.get("cuda_visible_devices") != allocation["cuda_visible_devices"]
            or pack_allocation.get("gpu_uuids") != [row["gpu_uuid"] for row in allocation["devices"]]
            or not same_allocation(allocation, pack_gate_allocation)
            or not any(source_matches(repo, source, pack_gate_ref) for source in metrics.get("sources", []))):
        raise ValueError("Nested pack is not the fixed same-allocation gate failure with linked provenance")
    # This is the actual audited pack object/command, not a synthesized gate.
    pack_payload = {**payload, "name": pack_name, "command": command}
    reason = validate_black_retry(pack_payload, job_id, fields, pack, pack_gate, pack_name + "_gate", "pack")
    return "probe_pack_nested_" + reason, {"initial_gate": gate_reference, "probe": probe_ref,
                                         "pack": pack_ref, "pack_gate": pack_gate_ref}

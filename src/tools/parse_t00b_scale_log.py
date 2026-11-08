"""Parse scale timing/resources and ordered fresh-RGB/phase evidence for 2/4/8 GPUs.

Invoked by the logged T00b wrapper, including on failure. Missing measurements
remain null; phase/counter inconsistency makes acceptance fail explicitly.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
TIMING = re.compile(r"timing_s/([A-Za-z_]+):(" + NUMBER + r")")
STEP = re.compile(r"\bstep:(\d+)\b")
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
REQUIRED_TIMINGS = ("step", "gen", "ref", "old_log_prob", "values", "update_critic", "update_actor")


def read_steps(text):
    result = {}
    for number, raw in enumerate(text.splitlines(), 1):
        line = ANSI.sub("", raw)
        match = STEP.search(line)
        timing = {key: float(value) for key, value in TIMING.findall(line)}
        if match and "step" in timing and all(math.isfinite(v) and v >= 0 for v in timing.values()):
            step = int(match.group(1))
            if step in result and result[step]["timing_s"] != timing:
                raise ValueError(f"Conflicting timing records for step {step}")
            result[step] = {"line": number, "timing_s": timing}
    return result


def read_jsonl(path):
    rows, errors = [], []
    if not path.exists():
        return rows, [f"Missing evidence: {path}"]
    for line, text in enumerate(path.read_text().splitlines(), 1):
        try:
            row = json.loads(text)
            if not isinstance(row, dict):
                raise ValueError("expected object")
            rows.append(row)
        except ValueError as error:
            errors.append(f"{path.name}:{line}: {error}")
    return rows, errors


def frame_summary(rows, completed_steps, hook_sha256, gpu_uuids=None):
    errors, by_step, other, began, ended = [], {}, Counter(), set(), set()
    scope = {"phase": "initialization", "training_step": None, "validation_step": None}
    stack, last_id, server_pid, sessions = [], 0, None, 0
    cached, rgb, startup_rgb, startup_black = 0, 0, 0, 0
    for row in rows:
        event_id = row.get("event_id")
        if type(event_id) is not int or event_id != last_id + 1:
            errors.append(f"Noncontiguous event_id after {last_id}: {event_id}")
        if type(event_id) is int:
            last_id = event_id
        if server_pid is None:
            server_pid = row.get("server_pid")
        if server_pid is None or row.get("server_pid") != server_pid:
            errors.append("Frame event writer identity changed or is missing")
        kind = row.get("kind")
        if kind == "session_start":
            sessions += 1
            if event_id != 1 or row.get("hook_sha256") != hook_sha256:
                errors.append("Unexpected session start or instrumentation hash")
        elif kind == "phase":
            operation, step = row.get("operation"), row.get("step")
            if row.get("previous_scope") != scope:
                errors.append(f"Phase previous-scope mismatch at event {event_id}")
            if operation == "train_begin":
                if type(step) is not int or step < 1 or step in began or scope["training_step"] is not None or stack:
                    errors.append(f"Invalid train_begin {step}")
                began.add(step)
                scope = {"phase": "training", "training_step": step, "validation_step": None}
                by_step.setdefault(step, Counter())
            elif operation == "train_end":
                if step not in began or step in ended or scope != {"phase": "training", "training_step": step, "validation_step": None} or stack:
                    errors.append(f"Invalid train_end {step}")
                ended.add(step)
                scope = {"phase": "idle", "training_step": None, "validation_step": None}
            elif operation == "validation_begin":
                stack.append(dict(scope))
                scope = {"phase": "validation", "training_step": scope["training_step"], "validation_step": step}
            elif operation == "validation_end":
                if not stack or scope["validation_step"] != step:
                    errors.append("Unmatched validation_end")
                else:
                    scope = stack.pop()
            else:
                errors.append(f"Unknown phase operation {operation}")
            if row.get("failed"):
                errors.append(f"Phase failed at event {event_id}")
        elif kind == "invalid_rgb":
            errors.append(f"Invalid fresh RGB packet at event {event_id}: shape={row.get('rgb_shape')}, nonzero={row.get('rgb_any_nonzero')}")
        elif kind == "controller_ready":
            if row.get("rgb_valid") is not True or row.get("rgb_shape") != [255, 255, 3]:
                errors.append(f"Controller completed initialization with invalid RGB at event {event_id}")
        elif kind in ("rgb", "cached_metadata"):
            if row.get("scope") != scope:
                errors.append(f"Packet phase mismatch at event {event_id}")
            if not isinstance(row.get("active_requests"), int) or row["active_requests"] < 1:
                errors.append(f"Packet outside a tracked environment request at event {event_id}")
            if not row.get("stream_id") or not isinstance(row.get("payload_bytes"), int) or row["payload_bytes"] < 1:
                errors.append(f"Invalid packet evidence at event {event_id}")
            if not re.fullmatch(r"[0-9a-f]{64}", str(row.get("payload_sha256"))):
                errors.append(f"Missing payload hash at event {event_id}")
            if gpu_uuids is not None and (row.get("gpu_identity") or {}).get("gpu_uuid") not in gpu_uuids:
                errors.append(f"Packet GPU identity is outside allocation at event {event_id}")
            if kind == "cached_metadata":
                cached += 1
                if row.get("field_type") != "METADATA_PATCH":
                    errors.append("Cached response marked with non-cache field type")
                continue
            field_type = row.get("field_type")
            initialization = row.get("controller_initialization") is True
            if (row.get("rgb_shape_valid") is not True or
                    (not initialization and (row.get("rgb_valid") is not True or row.get("rgb_any_nonzero") is not True))):
                errors.append(f"Fresh RGB packet lacks valid nonblack evidence at event {event_id}")
            if field_type not in ("RGB_IMAGE", "THIRD_PARTY_IMAGE"):
                errors.append(f"Non-RGB transport field counted at event {event_id}")
                continue
            rgb += 1
            startup_rgb += initialization
            startup_black += initialization and row.get("rgb_any_nonzero") is False
            camera = "main" if field_type == "RGB_IMAGE" else "third_party"
            step = scope["training_step"]
            if step is None:
                other[f"{scope['phase']}:{scope['validation_step']}:{camera}"] += 1
            else:
                target = by_step.setdefault(step, Counter())
                target[f"{scope['phase']}_{camera}"] += 1
                target[f"total_{camera}"] += 1
                target["constructor_rgb"] += initialization
                target["constructor_black_rgb"] += initialization and row.get("rgb_any_nonzero") is False
        else:
            errors.append(f"Unexpected event kind {kind!r}")
        if row.get("scope") != scope:
            errors.append(f"Recorded scope differs from reconstructed scope at event {event_id}")
    if sessions != 1 or rgb == 0:
        errors.append("Expected one server instrumentation session and observed real RGB packets")
    for step in completed_steps:
        if step not in began or step not in ended:
            errors.append(f"Completed timing step {step} lacks both explicit phase boundaries")
    missing_console = ended - set(completed_steps)
    if missing_console:
        errors.append(f"Phase-completed steps lack console timing: {sorted(missing_console)}")
    counts = {}
    for step in sorted(by_step):
        row = by_step[step]
        counts[str(step)] = {
            "training_main_rgb_frames": row["training_main"],
            "training_third_party_rgb_frames": row["training_third_party"],
            "validation_main_rgb_frames_within_step": row["validation_main"],
            "validation_third_party_rgb_frames_within_step": row["validation_third_party"],
            "total_new_rgb_frames": row["total_main"] + row["total_third_party"],
            "constructor_rgb_frames_in_total": row["constructor_rgb"],
            "constructor_black_rgb_frames_in_total": row["constructor_black_rgb"],
            "phase_begin_observed": step in began, "phase_end_observed": step in ended,
            "console_completed_update": step in completed_steps,
        }
    return {"valid": not errors, "errors": errors, "raw_event_count": len(rows),
            "new_rgb_packet_count_all_phases": rgb, "cached_metadata_response_count": cached,
            "constructor_rgb_packet_count_all_phases": startup_rgb,
            "constructor_black_rgb_packet_count_all_phases": startup_black,
            "steps": counts, "rgb_packets_outside_training_steps": dict(other),
            "last_scope": scope, "unclosed_validation_scopes": stack}


def median(values):
    return statistics.median(values) if values else None


def maximum(values):
    return max(values) if values else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("train-log", "resource-log", "frame-events", "manifest", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--archive-index", required=True, type=Path)
    parser.add_argument("--exit-code", type=int, required=True)
    parser.add_argument("--stage", required=True)
    args = parser.parse_args()
    warnings, config, steps = [], {}, {}
    archive, smoke = {}, {}
    try:
        archive = json.loads(args.archive_index.read_text())
        if not archive.get("complete"):
            warnings.append("Raw evidence archive is incomplete; scratch originals retained")
    except (OSError, ValueError) as error:
        warnings.append(f"Unreadable evidence archive index: {error}")
    if args.manifest.exists():
        try:
            config = json.loads(args.manifest.read_text())
        except ValueError as error:
            warnings.append(f"Unreadable manifest: {error}")
    gpu_count = config.get("allocation_gpu_count")
    smoke_path = Path(config.get("render_gate", {}).get("path", args.manifest.with_name("missing_render_gate.json")))
    try:
        gate_data = json.loads(smoke_path.read_text())
        smoke = gate_data.get("metrics", {})
        gate_meta = gate_data.get("_meta", {})
        if (gate_meta.get("producer") != "scripts/run_logged.py" or gate_meta.get("exit_code") != 0
                or hashlib.sha256(smoke_path.read_bytes()).hexdigest() != config.get("render_gate", {}).get("sha256")
                or gate_meta.get("metrics_sha256") != hashlib.sha256(json.dumps(smoke, sort_keys=True).encode()).hexdigest()
                or (gate_data.get("config", {}).get("allocation") or {}).get("pbs_jobid") != config.get("pbs_jobid")):
            smoke = {**smoke, "passed": False}
            warnings.append("Render gate provenance differs from preparation")
    except (OSError, ValueError) as error:
        warnings.append(f"Unreadable all-GPU render gate evidence: {error}")
    try:
        steps = read_steps(args.train_log.read_text(errors="replace") if args.train_log.exists() else "")
    except ValueError as error:
        warnings.append(str(error))
    resources, resource_errors = read_jsonl(args.resource_log)
    events, event_errors = read_jsonl(args.frame_events)
    warnings.extend(resource_errors + event_errors)
    frames = frame_summary(events, steps, config.get("hook_sha256"),
                           {device["gpu_uuid"] for device in config.get("allocation", {}).get("devices", [])})
    if event_errors:
        frames["valid"] = False
        frames["errors"].extend(event_errors)
    warnings.extend(frames["errors"])
    samples = [row for row in resources if row.get("type") == "sample"]
    completion = next((row for row in reversed(resources) if row.get("type") == "completion"), {})
    def values(key):
        return [row[key] for row in samples if type(row.get(key)) in (int, float) and math.isfinite(row[key])]
    timings = [steps[step]["timing_s"] for step in sorted(steps)]
    timing_keys = sorted(set(REQUIRED_TIMINGS).union(*(row.keys() for row in timings)))
    medians = {key: median([row[key] for row in timings if key in row]) for key in timing_keys}
    missing_timings = {str(step): [key for key in REQUIRED_TIMINGS if key not in row["timing_s"]]
                       for step, row in steps.items() if any(key not in row["timing_s"] for key in REQUIRED_TIMINGS)}
    if missing_timings:
        warnings.append(f"Completed steps missing required timing fields: {missing_timings}")
    devices = config.get("allocation", {}).get("devices", [])
    per_gpu = {}
    for device in devices:
        uuid = device["gpu_uuid"]
        observed = [row.get("gpus", {}).get(uuid, {}) for row in samples]
        def gpu_values(key):
            return [row[key] for row in observed if type(row.get(key)) in (int, float) and math.isfinite(row[key])]
        per_gpu[uuid] = {"logical_index": device["logical_index"],
                         "memory_peak_mib_sampled": maximum(gpu_values("memory_used_mib")),
                         "utilization_percent_median_sampled": median(gpu_values("utilization_percent")),
                         "valid_memory_samples": len(gpu_values("memory_used_mib"))}
    new_frames = ({str(step): frames["steps"][str(step)]["total_new_rgb_frames"] for step in steps}
                  if frames["valid"] else None)
    fps = ({str(step): new_frames[str(step)] / row["timing_s"]["step"]
            for step, row in steps.items() if row["timing_s"]["step"] > 0} if new_frames is not None else None)
    estimates = {}
    for target in (300, 100):
        wall = target * medians["step"] / 3600 if medians["step"] is not None else None
        estimates[str(target)] = {"configured_steps": target, "estimated_wall_hours": wall,
                                  "estimated_gpu_hours": wall * gpu_count if wall is not None and gpu_count in (2, 4, 8) else None,
                                  "gpu_hours_kind": "allocation_gpu_hours_before_scheduler_charge_multiplier",
                                  "allocation_gpus": gpu_count, "completed_steps_used": len(steps)}
    metrics = {
        "completed_training_steps": len(steps), "median_timing_seconds": medians,
        "median_step_seconds": medians["step"], "median_rollout_seconds": medians["gen"],
        "median_parameter_update_seconds": median([row["update_actor"] + row["update_critic"] for row in timings
                                                   if "update_actor" in row and "update_critic" in row]),
        "median_rollout_fraction_of_step": median([row["gen"] / row["step"] for row in timings if "gen" in row and row["step"] > 0]),
        "median_update_fraction_of_step": median([(row["update_actor"] + row["update_critic"]) / row["step"] for row in timings
                                                  if "update_actor" in row and "update_critic" in row and row["step"] > 0]),
        "gpu_resources_by_uuid": per_gpu, "process_tree_rss_peak_bytes": maximum(values("rss_bytes")),
        "process_tree_cpu_percent_median": median(values("cpu_percent")),
        "process_tree_cpu_percent_peak": maximum(values("cpu_percent")), "resource_sample_count": len(samples),
        "gpu_sample_error_count": sum("gpu_sample_error" in row for row in samples),
        "training_command_wall_seconds": completion.get("elapsed_seconds"),
        "training_command_exit_code": completion.get("returncode"), "wrapper_exit_code": args.exit_code,
        "timed_out": completion.get("timed_out"), "watchdog_seconds": completion.get("watchdog_seconds"),
        "cleanup_remaining": completion.get("cleanup_remaining"),
        "pretraining_rgb_smoke_passed": smoke.get("passed"),
        "actual_new_rgb_frames_per_training_step": new_frames,
        "required_render_fps_per_training_step": fps,
        "render_frame_accounting": frames, "full_training_estimates": estimates,
    }
    required = config.get("required_completed_updates")
    contiguous = sorted(steps) == list(range(1, len(steps) + 1))
    passed = (args.exit_code == 0 and completion.get("returncode") == 0 and smoke.get("passed") is True
              and type(required) is int and len(steps) >= required and contiguous
              and frames["valid"] and not missing_timings and gpu_count in (2, 4, 8) and len(per_gpu) == gpu_count
              and frames["last_scope"]["phase"] == "idle" and not frames["unclosed_validation_scopes"]
              and all(row["valid_memory_samples"] > 0 for row in per_gpu.values())
              and metrics["process_tree_rss_peak_bytes"] is not None and not resource_errors)
    if not passed:
        warnings.append("T00b acceptance not met: exit 0, required completed updates, complete timings, fresh-frame accounting and every allocated GPU resource required")
    evidence = []
    for path in (args.train_log, args.resource_log, args.frame_events, args.manifest, args.archive_index, smoke_path):
        if path.exists():
            evidence.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    archive_valid = bool(archive.get("complete"))
    for path in (args.train_log, args.resource_log, args.frame_events):
        recorded = archive.get("files", {}).get(path.name, {})
        actual = next((item for item in evidence if item["path"] == str(path)), {})
        if (not recorded.get("complete") or not actual
                or recorded.get("source_sha256") != actual.get("sha256")
                or recorded.get("source_bytes") != path.stat().st_size):
            archive_valid = False
            warnings.append(f"Archive/source identity could not be verified: {path.name}")
    if not archive_valid:
        passed = False
    result = {"metrics": metrics, "config": config, "acceptance_passed": passed,
              "last_stage": args.stage, "completed_step_timings": {str(k): v for k, v in steps.items()},
              "warnings": warnings, "evidence_files": evidence, "evidence_archive": archive,
              "pretraining_rgb_smoke": smoke,
              "measurement_notes": [
                  "Only step:<id> plus timing_s/step console records count as completed updates; raw values are rounded upstream.",
                  "RGB counts are fresh FIFO RGB_IMAGE and THIRD_PARTY_IMAGE payloads; metadata-patch cached images are excluded.",
                  "Fresh-frame totals include separately reported constructor startup overhead; black startup packets are not valid observations. Controller return frames and all later packets must be nonblack RGB.",
                  "Step total includes validation renders performed inside the step; separate training/validation/main/third-party counts remain available.",
                  "Explicit phase barriers reject outstanding server requests; incomplete steps and all initialization/final-validation events are retained.",
                  "required_render_fps is measured fresh step RGB frames / measured step seconds, not model turns or pure simulator benchmark FPS.",
                  "GPU peaks are sampled separately by verified allocation UUID; no sum of nonsimultaneous peaks is claimed.",
                  "RSS sums observed training/server processes and may double-count shared pages; compare PBS resources_used.mem separately.",
                  "300/100-step wall-hour estimates = target configured steps * observed median step seconds / 3600; Allocation GPU-hours multiply by the verified GPU count; scheduler charge multipliers are separate.",
                  "Estimates are conditional on this variant and observed completed steps, exclude initialization and unobserved later validation/checkpoint costs, and are not completed training.",
                  "PBS allocation time and charges must be read from final scheduler records separately."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print("T00B_PARSE_STATUS: " + json.dumps({"acceptance_passed": passed, "warnings": warnings}), flush=True)
    print("FINAL_METRICS_JSON: " + json.dumps(metrics, sort_keys=True, allow_nan=False), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

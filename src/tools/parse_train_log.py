#!/usr/bin/env python3
"""Extract completed official VAGEN steps and observed T00 resources, including failures."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
TIMING = re.compile(r"timing_s/([A-Za-z_]+):(" + NUMBER + r")")
NAVIGATION_TURNS = re.compile(
    r"(?<!\S)(train/step/NavigationEnvConfig\([^\r\n)]*\)):(" + NUMBER + r")(?= - |$)"
)
STEP = re.compile(r"\bstep:(\d+)\b")
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def read_steps(text):
    """A 'global_steps' announcement is not a completed optimizer update."""
    steps = {}
    for line_number, raw in enumerate(text.splitlines(), 1):
        line = ANSI.sub("", raw)
        match = STEP.search(line)
        values = {key: float(value) for key, value in TIMING.findall(line)}
        if match and "step" in values and all(math.isfinite(v) and v >= 0 for v in values.values()):
            turns = {key: float(value) for key, value in NAVIGATION_TURNS.findall(line)
                     if math.isfinite(float(value)) and float(value) >= 0}
            steps[int(match.group(1))] = {
                "line": line_number, "timing_s": values, "navigation_mean_model_turns": turns,
            }
    return steps


def median(values):
    return statistics.median(values) if values else None


def maximum(values):
    return max(values) if values else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("train-log", "resource-log", "manifest", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--exit-code", required=True, type=int)
    parser.add_argument("--stage", required=True)
    args = parser.parse_args()
    warnings = []
    config = {}
    if args.manifest.exists():
        try:
            config = json.loads(args.manifest.read_text())
        except (OSError, ValueError) as error:
            warnings.append(f"Unreadable config manifest: {error}")
    text = args.train_log.read_text(errors="replace") if args.train_log.exists() else ""
    steps = read_steps(text)
    timings = [entry["timing_s"] for _, entry in sorted(steps.items())]
    samples, completion = [], {}
    if args.resource_log.exists():
        for number, line in enumerate(args.resource_log.read_text(errors="replace").splitlines(), 1):
            try:
                row = json.loads(line)
                if row.get("type") == "sample":
                    samples.append(row)
                elif row.get("type") == "completion":
                    completion = row
            except ValueError:
                warnings.append(f"Unreadable resource line {number}")
    field = lambda key: [row[key] for row in samples if isinstance(row.get(key), (float, int)) and math.isfinite(row[key])]
    durations = [row["step"] for row in timings]
    rollout = [row["gen"] for row in timings if "gen" in row]
    updates = [row["update_actor"] + row["update_critic"] for row in timings
               if "update_actor" in row and "update_critic" in row]
    metrics = {
        "completed_training_steps": len(steps),
        "median_step_seconds": median(durations),
        "median_rollout_seconds": median(rollout),
        "median_parameter_update_seconds": median(updates),
        "median_rollout_fraction_of_step": median([row["gen"] / row["step"] for row in timings
                                                   if "gen" in row and row["step"] > 0]),
        "median_update_fraction_of_step": median([(row["update_actor"] + row["update_critic"]) / row["step"]
                                                  for row in timings if "update_actor" in row
                                                  and "update_critic" in row and row["step"] > 0]),
        "median_actor_update_seconds": median([row["update_actor"] for row in timings if "update_actor" in row]),
        "median_critic_update_seconds": median([row["update_critic"] for row in timings if "update_critic" in row]),
        "gpu_memory_peak_mib_sampled": maximum(field("gpu_memory_used_mib")),
        "process_tree_cpu_percent_median": median(field("cpu_percent")),
        "process_tree_cpu_percent_peak": maximum(field("cpu_percent")),
        "process_tree_rss_peak_bytes": maximum(field("rss_bytes")),
        "resource_sample_count": len(samples),
        "training_command_wall_seconds": completion.get("elapsed_seconds"),
        "training_command_exit_code": completion.get("returncode"),
        "wrapper_exit_code": args.exit_code,
        "navigation_mean_model_turns_by_step": {
            str(step): entry["navigation_mean_model_turns"] for step, entry in sorted(steps.items())
        },
        "actual_new_rgb_frames_per_train_step": None,
        "required_render_fps": None,
    }
    official_steps = config.get("official_total_training_steps")
    if official_steps is not None:
        metrics["official_config_total_training_steps"] = official_steps
    metrics["estimated_full_training_hours"] = (
        official_steps * metrics["median_step_seconds"] / 3600
        if official_steps is not None and durations else None
    )
    for key in ("median_step_seconds", "median_rollout_seconds", "median_parameter_update_seconds",
                "gpu_memory_peak_mib_sampled", "process_tree_rss_peak_bytes"):
        if metrics[key] is None:
            warnings.append(f"Not observed: {key}; no substitute value was generated")
    passed = args.exit_code == 0 and len(steps) >= 20
    if not passed:
        warnings.append("T00 training acceptance not met: requires exit 0 and at least 20 completed updates")
    sources = []
    for path in (args.train_log, args.resource_log, args.manifest):
        if path.exists():
            sources.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    result = {
        "metrics": metrics, "config": config, "kind": "T00_official_Navigation_feasibility",
        "acceptance_passed": passed, "last_stage": args.stage, "warnings": warnings,
        "evidence_files": sources, "completed_step_timings": {str(k): v for k, v in sorted(steps.items())},
        "measurement_notes": [
            "Only console records containing step:<id> and timing_s/step count as completed updates.",
            "Official console timings are rounded to three decimals; gen includes rollout and remote rendering.",
            "Navigation train/step scalars are rounded per-configuration mean model turns, retained separately with source lines in completed_step_timings; no weighted global mean is inferred.",
            "Actual new RGB frames and required render FPS are unmeasured: turns can execute zero to five actions and return a cached frame, reset adds transactions, and transactions do not necessarily produce new images. These counts cannot be recovered from the logged scalar means.",
            "Update time is update_actor + update_critic for the same step; other forward/reward/validation costs remain separate.",
            "Resource peaks are sampled, not guaranteed instantaneous peaks. GPU memory covers the sole allocated board.",
            "CPU 100% means one core; RSS is summed over observed training/server descendants, including shared pages.",
            "Full-training estimate = official configured steps * observed median step seconds / 3600; excludes setup/startup and extra final validation/save overhead.",
            "Training command wall time is not PBS charged walltime; account total GPU-hours from scheduler records.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print("T00_PARSE_STATUS: " + json.dumps({"acceptance_passed": passed, "warnings": warnings}), flush=True)
    print("FINAL_METRICS_JSON: " + json.dumps(metrics, sort_keys=True, allow_nan=False), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

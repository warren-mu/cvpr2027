"""Summarize explicit T00b 2/4/8-GPU runs; execute via run_logged after commit.

Declare all six slots with repeated --baseline N=results/file.json and
--variant N=results/file.json. Supply every other attempt via --history and
final PBS producers via --accounting. A declared absent file stays missing;
history is never searched for a replacement. No scheduler or GPU operations.
"""
import argparse
import copy
import importlib.util
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase_progress as provenance
from t00_summary import accounting_summary
from src.envs.thor_common import write_result

GPU_COUNTS = (2, 4, 8)
VARIANTS = ("baseline", "vllm04")
MEMORY_FIELDS = ("Resource_List.walltime", "Resource_List.mem", "resources_used.mem", "resources_used.vmem",
                 "Resource_List.select", "Resource_List.nodes", "Resource_List.nodect",
                 "exec_vnode", "Resource_List.queue_charge_rate")
NOTES = [
    "Explicit slots only; no latest, fastest, successful-only search, replacement from history, or pooled medians.",
    "Program success means evidence processing completed, not scientific acceptance or final T00b acceptance.",
    "Failure measurements remain in raw_source_record; ineligible comparisons and conditional estimates stay null.",
    "Speedup=t_2/t_N; efficiency=speedup/(N/2), using source median_step_seconds of the same variant.",
    "Approved 2-GPU TP1 versus 4/8-GPU TP4 changes topology, CPU/RAM allocation and renderer concurrency; this is not pure hardware strong scaling.",
    "Variant gen ratio=baseline_gen/vllm04_gen; delta_seconds=vllm04_gen-baseline_gen. Actual observed window lengths are retained, usually 20 versus 5 updates.",
    "100/300 estimates are copied unchanged from each parser: configured_steps * observed median step seconds / 3600, then multiply by allocation GPUs. No legacy-step correction or recomputation.",
    "Those conditional estimates exclude initialization and unobserved later validation/checkpoint costs; allocation GPUh are before scheduler charge multipliers.",
    "No estimate of 20 main-training runs, completion date, general queue-wait distribution, or recommended GPU count is inferred.",
    "Fresh RGB, validation/constructor attribution, timings and per-UUID sampled peaks are copied; peaks are not summed across GPUs and required_render_fps is not peak simulator FPS.",
    "Process-tree RSS can double-count shared pages and is separate from PBS resources_used.mem. PBS memory fields remain raw; no unit conversion or threshold comparison is performed.",
    "Final accounting deduplicates full PBS IDs, including failures. Baseline and variant in one allocation link to the same record. All explicitly supplied accounting jobs are included, with unrelated job IDs disclosed.",
    "A successful baseline is not invalidated by a later variant's failure or the resulting nonzero whole-PBS exit. Whole-job usage is not apportioned to child runs by command duration.",
    "Accounting is allocation GPUh, not a reconciled credit balance or a replacement for the submit-time 60-GPUh charged-budget guard.",
    "Infrastructure/packed-task smoke can remain explicit history, but is never treated as VAGEN training throughput.",
]


def source_path(requested, output):
    path = (ROOT / requested).resolve()
    relative = path.relative_to(ROOT)
    if len(relative.parts) < 2 or relative.parts[0] != "results" or path.suffix != ".json" or path == output:
        raise ValueError("Require a source results JSON distinct from output")
    return path, relative.as_posix()


def parse_slots(baselines, variants):
    slots = {}
    for variant, declarations in (("baseline", baselines), ("vllm04", variants)):
        for declaration in declarations:
            number, separator, requested = declaration.partition("=")
            if not separator or number not in ("2", "4", "8") or not requested:
                raise ValueError("Use explicit N=results/file.json slots, N in 2/4/8")
            key = f"{variant}_{number}"
            if key in slots:
                raise ValueError(f"Duplicate slot: {key}")
            slots[key] = {"variant": variant, "gpus": int(number), "requested_source": requested}
    if set(slots) != {f"{variant}_{count}" for variant in VARIANTS for count in GPU_COUNTS}:
        raise ValueError("Declare all six slots, including paths for results not yet produced")
    return slots


def load_inputs(requested, output, auditor, sources=None, audits=None, missing=None):
    # Caller-owned evidence lists retain already verified references if a later
    # source is corrupt. No partial scientific summary is emitted in that case.
    sources = sources if sources is not None else []
    audits = audits if audits is not None else []
    missing = missing if missing is not None else []
    loaded, identities = {}, set()
    for name in requested:
        path, relative = source_path(name, output)
        if relative in loaded:
            raise ValueError(f"Duplicate explicit source path: {relative}")
        if not path.exists():
            loaded[relative] = (None, None)
            missing.append({"requested_source": relative, "sha256": None, "reason": "explicit source file not produced or unavailable"})
            continue
        data, reference, audit = provenance.load_source(path, output, auditor)
        json.dumps(data, allow_nan=False)  # Non-finite JSON cannot be preserved as valid raw evidence.
        meta = data["_meta"]
        identity = (meta.get("name"), meta.get("start"), meta.get("git_commit"))
        if identity in identities:
            raise ValueError(f"Duplicate source run identity: {relative}")
        identities.add(identity)
        loaded[relative] = (data, reference)
        sources.append(reference)
        audits.append(audit)
    return loaded, sources, audits, missing


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def job_id(data):
    config = data.get("config", {})
    return (config.get("pbs_jobid") or config.get("pbs_job_id")
            or (config.get("allocation") or {}).get("pbs_jobid")
            or (config.get("runtime") or {}).get("pbs_jobid"))


def eligibility(data, gpus, variant):
    if data is None:
        return ["explicit_source_missing"]
    metrics, config = data["metrics"], data["config"]
    reasons = []
    if data.get("acceptance_passed") is not True:
        reasons.append("original_acceptance_not_true")
    for label, code in (("meta", data["_meta"].get("exit_code")),
                        ("wrapper", metrics.get("wrapper_exit_code")),
                        ("training_command", metrics.get("training_command_exit_code"))):
        if type(code) is not int or code != 0:
            reasons.append(label + "_exit_not_zero")
    count = metrics.get("completed_training_steps")
    minimum = 20 if variant == "baseline" else 5
    if type(count) is not int or count < minimum:
        reasons.append("insufficient_completed_updates")
    if config.get("variant") != variant or config.get("baseline") != "official_vagen_base_navigation":
        reasons.append("not_the_declared_official_navigation_variant")
    if type(config.get("rollout_tensor_parallel_size")) is not int or config["rollout_tensor_parallel_size"] != (1 if gpus == 2 else 4):
        reasons.append("unexpected_or_missing_tensor_parallel_topology")
    if config.get("additional_critic_offload") is not False:
        reasons.append("additional_critic_offload_not_verified_false")
    allocation = config.get("allocation") or {}
    devices = allocation.get("devices")
    resources = metrics.get("gpu_resources_by_uuid")
    identities = [row.get("gpu_uuid") for row in devices] if isinstance(devices, list) and all(isinstance(row, dict) for row in devices) else []
    if (len(identities) != gpus or any(not isinstance(value, str) or not value for value in identities)
            or len(set(identities)) != gpus or not isinstance(resources, dict) or set(resources) != set(identities)):
        reasons.append("allocation_and_sampled_gpu_uuid_sets_do_not_match_slot")
    # Old two-GPU and new scale schemas share the identities above; the optional
    # explicit count is checked only when present, never synthesized as evidence.
    if "allocation_gpu_count" in config and (type(config["allocation_gpu_count"]) is not int or config["allocation_gpu_count"] != gpus):
        reasons.append("declared_gpu_count_mismatch")
    if not isinstance(job_id(data), str) or not re.fullmatch(r"\d+\.hopper-m-\d+", job_id(data)) or allocation.get("pbs_jobid") != job_id(data):
        reasons.append("allocation_job_identity_missing_or_mismatched")
    if not positive(metrics.get("median_step_seconds")):
        reasons.append("median_step_not_finite_positive")
    return reasons


def run_record(data, reference, requested):
    return {"requested_source": requested, "source": reference,
            "status": "missing" if data is None else "recorded",
            "pbs_jobid": job_id(data) if data else None,
            "raw_source_record": copy.deepcopy(data),
            "final_accounting": None,
            "accounting_unavailable_reason": "not linked yet"}


def variant_pair_reasons(baseline, variant):
    if not baseline["eligible"] or not variant["eligible"]:
        return ["both_selected_runs_must_be_eligible"]
    base, changed = baseline["raw_source_record"], variant["raw_source_record"]
    reference = changed["config"].get("same_job_baseline") or {}
    reasons = []
    if job_id(base) != job_id(changed):
        reasons.append("variant_and_baseline_have_different_pbs_jobs")
    recorded_path = reference.get("path")
    if not isinstance(recorded_path, str) or (ROOT / recorded_path).resolve() != (ROOT / baseline["source"]["file"]).resolve():
        reasons.append("variant_baseline_path_mismatch")
    if reference.get("sha256") != baseline["source"]["sha256"]:
        reasons.append("variant_baseline_full_sha_mismatch")
    for key in ("vagen_commit", "verl_commit", "rollout_tensor_parallel_size", "additional_critic_offload"):
        if base["config"].get(key) is None or base["config"].get(key) != changed["config"].get(key):
            reasons.append("variant_configuration_mismatch:" + key)
    return reasons


def comparisons(slots):
    scaling = []
    for variant in VARIANTS:
        base = slots[f"{variant}_2"]
        for count in (4, 8):
            target = slots[f"{variant}_{count}"]
            reasons = []
            if not base["eligible"] or not target["eligible"]:
                reasons.append("both_explicit_slots_must_be_eligible")
            else:
                for key in ("vagen_commit", "verl_commit"):
                    value = base["raw_source_record"]["config"].get(key)
                    if not value or value != target["raw_source_record"]["config"].get(key):
                        reasons.append("source_revision_mismatch:" + key)
            speedup = None if reasons else base["raw_source_record"]["metrics"]["median_step_seconds"] / target["raw_source_record"]["metrics"]["median_step_seconds"]
            scaling.append({"variant": variant, "reference_gpus": 2, "target_gpus": count,
                            "speedup": speedup, "efficiency": speedup / (count / 2) if speedup is not None else None,
                            "unavailable_reasons": reasons,
                            "scope": "approved topology comparison; TP1 versus TP4 and allocation/renderer changes, not pure hardware strong scaling"})
    variants = []
    for count in GPU_COUNTS:
        base, variant = slots[f"baseline_{count}"], slots[f"vllm04_{count}"]
        reasons = variant_pair_reasons(base, variant)
        base_metrics = (base["raw_source_record"] or {}).get("metrics", {})
        variant_metrics = (variant["raw_source_record"] or {}).get("metrics", {})
        left, right = base_metrics.get("median_rollout_seconds"), variant_metrics.get("median_rollout_seconds")
        if not positive(left) or not positive(right):
            reasons.append("gen_medians_not_finite_positive")
        variants.append({"gpus": count, "baseline_completed_updates": base_metrics.get("completed_training_steps"),
                         "vllm04_completed_updates": variant_metrics.get("completed_training_steps"),
                         "baseline_gen_over_vllm04_gen": None if reasons else left / right,
                         "vllm04_minus_baseline_gen_seconds": None if reasons else right - left,
                         "unavailable_reasons": reasons,
                         "window_note": "different observed update windows, normally baseline20 versus variant5; no paired-step or significance claim"})
    return {"scaling": scaling, "variant_generation": variants}


def link_accounting(inputs, runs, missing_accounting):
    summary = accounting_summary(inputs)
    extras = {}
    for data, reference in inputs:
        for row in data["config"]["jobs"]:
            fields = row.get("fields", {})
            if row.get("qstat_exit") != 0 or fields.get("job_state") != "F":
                continue
            values = {key: fields.get(key) for key in MEMORY_FIELDS}
            if row["job_id"] in extras and extras[row["job_id"]] != values:
                raise ValueError("Conflicting final memory/select/charge fields: " + row["job_id"])
            extras[row["job_id"]] = values
    by_id = {row["job_id"]: row for row in summary["jobs"]}
    for identity, row in by_id.items():
        row["raw_scheduler_fields"] = extras.get(identity)
        row["memory_within_requested"] = None
        row["memory_comparison_unavailable_reason"] = "raw scheduler values preserved; no memory unit conversion or threshold comparison performed"
    expected = {row["pbs_jobid"] for row in runs if row["pbs_jobid"]}
    missing = expected - set(by_id)
    for row in runs:
        final = by_id.get(row["pbs_jobid"])
        if (final is not None and "gpus" in row and final["allocated_ngpus"] is not None
                and final["allocated_ngpus"] != row["gpus"]):
            raise ValueError("Selected training slot disagrees with final allocated GPU count: " + row["pbs_jobid"])
        row["final_accounting"] = copy.deepcopy(final)
        row["accounting_scope"] = "whole PBS allocation; shared by child runs and never apportioned"
        row["accounting_unavailable_reason"] = None if final else "source missing, PBS ID unavailable, or no supplied verified final accounting"
    complete = bool(inputs) and not missing_accounting and not missing and not summary["unresolved_job_ids"]
    summary.update(complete_for_supplied_run_jobs=complete,
                   run_job_ids_without_final_accounting=sorted(missing),
                   accounting_job_ids_not_linked_to_runs=sorted(set(by_id) - expected),
                   missing_accounting_sources=missing_accounting,
                   total_scope="all explicitly supplied accounting jobs, including disclosed jobs not linked to these runs; not whole-project budget")
    if not complete:
        summary["total_allocation_gpu_hours"] = None
    if not inputs:
        summary["known_allocation_gpu_hours_subtotal"] = None
    return summary


def summarize(declarations, history, accounting, loaded, output):
    slots, history_rows = {}, []
    for key, declaration in declarations.items():
        _, requested = source_path(declaration["requested_source"], output)
        data, reference = loaded[requested]
        row = run_record(data, reference, requested)
        reasons = eligibility(data, declaration["gpus"], declaration["variant"])
        row.update(gpus=declaration["gpus"], variant=declaration["variant"], eligible=not reasons,
                   eligibility_reasons=reasons, eligible_estimates=None)
        slots[key] = row
    # A variant is not eligible for reporting without its explicitly chosen,
    # same-allocation successful baseline; never substitute another attempt.
    for count in GPU_COUNTS:
        row = slots[f"vllm04_{count}"]
        reasons = variant_pair_reasons(slots[f"baseline_{count}"], row)
        row["eligibility_reasons"].extend(reasons)
        row["eligible"] = not row["eligibility_reasons"]
    for row in slots.values():
        estimates = ((row["raw_source_record"] or {}).get("metrics", {}).get("full_training_estimates"))
        if row["eligible"] and isinstance(estimates, dict):
            row["eligible_estimates"] = {key: copy.deepcopy(estimates.get(key)) for key in ("100", "300")}
        row["estimates_unavailable_reason"] = (None if row["eligible_estimates"] is not None else
                                                 "run ineligible or source parser estimates unavailable; not recomputed")
        row["estimate_target_missing_reasons"] = {key: "source parser target unavailable; not recomputed"
            for key in ("100", "300") if row["eligible"] and (not isinstance(estimates, dict) or estimates.get(key) is None)}
    for name in history:
        _, requested = source_path(name, output)
        data, reference = loaded[requested]
        row = run_record(data, reference, requested)
        row["role"] = "history_only; never selected for comparison or used to fill a missing slot"
        history_rows.append(row)
    account_inputs, missing_accounting = [], []
    for name in accounting:
        _, requested = source_path(name, output)
        data, reference = loaded[requested]
        if data is None:
            missing_accounting.append(requested)
        else:
            account_inputs.append((data, reference))
    accounts = link_accounting(account_inputs, [*slots.values(), *history_rows], missing_accounting)
    all_eligible = all(row["eligible"] for row in slots.values())
    history_missing = any(row["status"] == "missing" for row in history_rows)
    return {"aggregation_program_completed": True,
            "research_status": "six_selected_runs_eligible" if all_eligible else "incomplete",
            "all_six_selected_runs_eligible": all_eligible,
            "final_task_acceptance": None,
            "final_task_acceptance_reason": "this producer does not judge D/E/F/G, packed smoke, human review, or full T00b acceptance",
            "evidence_status": "complete_for_explicit_inputs" if all_eligible and accounts["complete_for_supplied_run_jobs"] and not history_missing else "incomplete",
            "slots": slots, "history": history_rows, "comparisons": comparisons(slots),
            "accounting": accounts, "errors": []}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--baseline", action="append", required=True)
    parser.add_argument("--variant", action="append", required=True)
    parser.add_argument("--history", nargs="+", action="extend", default=[])
    parser.add_argument("--accounting", nargs="+", action="extend", default=[])
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_]+", args.name):
        parser.error("Use a unique safe run name")
    output = ROOT / "results" / (args.name + ".json")
    if output.exists():
        raise FileExistsError("Existing summary is preserved; use a unique name")
    sources, audits, missing = [], [], []
    metrics = {"aggregation_program_completed": False, "research_status": "unassessed",
               "all_six_selected_runs_eligible": None, "final_task_acceptance": None, "errors": []}
    config = {"name": args.name, "requested_inputs": vars(args), "notes": NOTES,
              "source_provenance_checks": audits, "missing_sources": missing}
    try:
        declarations = parse_slots(args.baseline, args.variant)
        requested = [row["requested_source"] for row in declarations.values()] + args.history + args.accounting
        spec = importlib.util.spec_from_file_location("training_summary_existing_auditor", ROOT / "scripts/audit_provenance.py")
        auditor = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(auditor)
        loaded, _, _, _ = load_inputs(requested, output, auditor, sources, audits, missing)
        metrics = summarize(declarations, args.history, args.accounting, loaded, output)
    except (OSError, ValueError, KeyError, TypeError) as error:
        metrics["errors"].append(f"{type(error).__name__}: {error}")
    if output.exists():
        raise FileExistsError("Output appeared during aggregation; refusing replacement")
    write_result(args.name, metrics, config)
    output.write_text(json.dumps({"metrics": metrics, "config": config, "sources": sources},
                                 indent=2, allow_nan=False) + "\n")
    return 0 if metrics["aggregation_program_completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

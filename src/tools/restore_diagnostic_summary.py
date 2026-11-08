#!/usr/bin/env python3
"""Summarize the three saved restore diagnostics, without running the simulator.

Run with: python3 scripts/run_logged.py --name t00_restore_diagnostic_summary -- \
    python3 src/tools/restore_diagnostic_summary.py
Aggregation success is independent of the source RGB-equality outcomes.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import statistics


ROOT = Path(__file__).resolve().parents[2]
MODES = ("no_reset", "metadata", "paused")
EXACT_FIELDS = ("object_inventory_equal", "all_object_position_rotation_exact",
                "agent_fields_exact", "camera_reported_fields_exact",
                "observed_transforms_exact", "observed_state_fields_exact")
DELTA_FIELDS = ("max_object_position_delta_m", "max_object_raw_euler_component_delta_degrees",
                "max_object_wrapped_euler_component_delta_degrees")


def metrics_hash(metrics):
    return hashlib.sha256(json.dumps(metrics, sort_keys=True, allow_nan=False).encode()).hexdigest()


def validate_document(data, mode):
    if not isinstance(data, dict) or not isinstance(data.get("metrics"), dict) or not isinstance(data.get("config"), dict):
        raise ValueError("Expected metrics and config objects")
    meta = data.get("_meta")
    if not isinstance(meta, dict) or meta.get("producer") != "scripts/run_logged.py":
        raise ValueError("Missing run_logged provenance")
    if type(meta.get("exit_code")) is not int:
        raise ValueError("Source run has no final integer exit code")
    if meta.get("metrics_sha256") != metrics_hash(data["metrics"]):
        raise ValueError("Source metrics_sha256 mismatch")
    if data["config"].get("mode") != mode or data["config"].get("cases") != 20:
        raise ValueError(f"Expected mode={mode} with 20 configured cases")


def measured(value, label, integer=False):
    if value is None:
        return None
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            or value < 0 or (integer and type(value) is not int)):
        raise ValueError(f"{label}: expected finite nonnegative {'integer' if integer else 'number'} or null")
    return value


def stats(values):
    observed = [value for value in values if value is not None]
    return {"observed_count": len(observed), "min": min(observed) if observed else None,
            "median": statistics.median(observed) if observed else None,
            "max": max(observed) if observed else None}


def truth_counts(values):
    if any(value is not None and type(value) is not bool for value in values):
        raise ValueError("Exact-comparison values must be boolean or null")
    return {"true": sum(value is True for value in values),
            "false": sum(value is False for value in values),
            "unknown": sum(value is None for value in values)}


def compact_comparison(row):
    if row is None:
        return None
    if not isinstance(row, dict):
        raise ValueError("Transform comparison must be an object")
    keys = (*EXACT_FIELDS, *DELTA_FIELDS, "objects_compared", "missing_in_second", "new_in_second",
            "agent_field_changes", "camera_field_changes", "render_field_changes",
            "object_auxiliary_field_changes", "isSceneAtRest_values", "changed_object_transforms",
            "agent_position_delta_m", "camera_position_delta_m")
    return {key: copy.deepcopy(row[key]) for key in keys if key in row}


def comparison_summary(rows):
    summary = {"comparisons_expected": len(rows), "comparisons_available": sum(row is not None for row in rows)}
    summary["exact"] = {key: truth_counts([row.get(key) if row else None for row in rows]) for key in EXACT_FIELDS}
    summary["deltas"] = {key: stats([measured(row.get(key), key) if row else None for row in rows]) for key in DELTA_FIELDS}
    for key in ("agent_position_delta_m", "camera_position_delta_m"):
        summary["deltas"][key + "_l2"] = stats([
            measured(row[key].get("l2"), key) if row and isinstance(row.get(key), dict) else None for row in rows])
    return summary


def summarize_mode(data, mode):
    validate_document(data, mode)
    metrics, config = data["metrics"], data["config"]
    resolution = measured(config.get("resolution"), "resolution", integer=True)
    if not resolution:
        raise ValueError("Missing positive resolution")
    area = resolution ** 2
    cases = metrics.get("cases")
    if not isinstance(cases, list):
        raise ValueError("Missing case observations")
    by_index = {}
    for row in cases:
        index = row.get("state_index")
        if type(index) is not int or not 0 <= index < 20 or index in by_index:
            raise ValueError("Duplicate or invalid state_index")
        by_index[index] = row
    case_rows, histograms, edge_rows = [], [], []
    pairs, source_comparisons, before_first = [], [], []
    for index in range(20):
        row = by_index.get(index, {})
        rgb = row.get("rgb")
        compact = {"state_index": index, "status": row.get("status", "not_observed"),
                   "random_seed": row.get("random_seed"), "error": row.get("error"),
                   "evidence_error": row.get("evidence_error"), "rgb": None}
        if rgb is not None:
            if not isinstance(rgb, dict):
                raise ValueError("RGB observation must be an object")
            pixels = measured(rgb.get("different_pixels"), "different_pixels", integer=True)
            channels = measured(rgb.get("different_channel_values"), "different_channel_values", integer=True)
            amplitude = measured(rgb.get("max_pixel_difference"), "max_pixel_difference", integer=True)
            if pixels is not None and pixels > area or channels is not None and channels > area * 3 or amplitude is not None and amplitude > 255:
                raise ValueError("RGB measurement exceeds image bounds")
            if pixels is not None and channels is not None and not pixels <= channels <= 3 * pixels:
                raise ValueError("Differing-pixel and channel counts disagree")
            if pixels is not None and amplitude is not None and (pixels == 0) != (amplitude == 0):
                raise ValueError("Pixel-difference count and maximum amplitude disagree")
            exact = rgb.get("rgb_exact_zero")
            truth_counts([exact])
            if pixels is not None and exact is not None and exact != (pixels == 0):
                raise ValueError("RGB exact-zero flag contradicts differing-pixel count")
            histogram = rgb.get("absolute_channel_difference_histogram")
            if histogram is not None:
                if not isinstance(histogram, list) or len(histogram) != 256:
                    raise ValueError("Expected 256 absolute channel-difference bins")
                histogram = [measured(value, "histogram bin", integer=True) for value in histogram]
                if any(value is None for value in histogram) or sum(histogram) != area * 3:
                    raise ValueError("Histogram does not cover all RGB channels")
                if channels is not None and sum(histogram[1:]) != channels:
                    raise ValueError("Histogram disagrees with different_channel_values")
                if amplitude is not None and max(i for i, count in enumerate(histogram) if count) != amplitude:
                    raise ValueError("Histogram disagrees with max_pixel_difference")
                histograms.append(histogram)
            compact["rgb"] = {key: copy.deepcopy(rgb.get(key)) for key in
                              ("rgb_exact_zero", "different_pixels", "different_channel_values", "max_pixel_difference",
                               "changed_bbox_xyxy_inclusive", "edge_analysis_available", "edge_band_radius_pixels")}
            if rgb.get("edge_analysis_available") is True:
                band = measured(rgb.get("edge_band_pixels"), "edge_band_pixels", integer=True)
                near = measured(rgb.get("changed_pixels_in_edge_band"), "changed_pixels_in_edge_band", integer=True)
                outside = measured(rgb.get("changed_pixels_outside_edge_band"), "changed_pixels_outside_edge_band", integer=True)
                if None in (pixels, band, near, outside) or band > area or near > band or outside > area - band or near + outside != pixels:
                    raise ValueError("Inconsistent edge-band counts")
                baseline = band / area
                fraction = near / pixels if pixels else None
                edge = {"state_index": index, "different_pixels": pixels, "edge_band_pixels": band,
                        "changed_pixels_in_edge_band": near, "changed_pixels_outside_edge_band": outside,
                        "edge_band_area_fraction": baseline, "changed_pixel_fraction_in_edge_band": fraction,
                        "observed_minus_uniform_spatial_baseline": fraction - baseline if fraction is not None else None,
                        "ratio_to_uniform_spatial_baseline": fraction / baseline if fraction is not None and baseline else None}
                edge_rows.append(edge)
                compact["edge"] = edge
        pair = compact_comparison(row.get("transforms"))
        sources = row.get("source_to_capture_transforms")
        if sources is not None and (not isinstance(sources, list) or len(sources) != 2):
            raise ValueError("Expected two source-to-capture comparisons or absent observations")
        sources = [compact_comparison(item) for item in sources] if sources is not None else [None, None]
        initial = compact_comparison(row.get("before_first_vs_first"))
        pairs.append(pair)
        source_comparisons.extend(sources)
        before_first.append(initial)
        compact.update(pair_transforms=pair, source_to_capture_transforms=sources, before_first_vs_first=initial)
        case_rows.append(compact)
    differences = [row["rgb"] for row in case_rows if row["rgb"] is not None]
    changed_total = sum(row["different_pixels"] for row in edge_rows)
    near_total = sum(row["changed_pixels_in_edge_band"] for row in edge_rows)
    baseline = sum(row["different_pixels"] * row["edge_band_area_fraction"] for row in edge_rows) / changed_total if changed_total else None
    fraction = near_total / changed_total if changed_total else None
    return {
        "source_status": metrics.get("status"), "source_exit_code": data["_meta"]["exit_code"],
        "source_diagnostic_completed": metrics.get("diagnostic_completed"),
        "source_exact_zero_passed": metrics.get("exact_zero_passed"),
        "source_error": metrics.get("error"), "source_cleanup_errors": metrics.get("cleanup_errors"),
        "expected_cases": 20, "observed_cases": len(cases),
        "completed_cases": sum(row.get("status") == "completed" for row in cases),
        "rgb_observed_cases": len(differences), "rgb_exact_zero": truth_counts([row["rgb"].get("rgb_exact_zero") if row["rgb"] else None for row in case_rows]),
        "rgb": {key: stats([row.get(key) for row in differences]) for key in
                ("different_pixels", "different_channel_values", "max_pixel_difference")},
        "absolute_channel_difference_histogram": [sum(bins) for bins in zip(*histograms)] if histograms else None,
        "histogram_cases": len(histograms),
        "edge": {"cases_with_counts": len(edge_rows),
                 "edge_band_area_fraction": stats([row["edge_band_area_fraction"] for row in edge_rows]),
                 "changed_pixel_fraction_in_edge_band": stats([row["changed_pixel_fraction_in_edge_band"] for row in edge_rows]),
                 "different_pixels_total": changed_total if edge_rows else None,
                 "changed_pixels_in_edge_band_total": near_total if edge_rows else None,
                 "changed_pixels_outside_edge_band_total": changed_total - near_total if edge_rows else None,
                 "pooled_changed_pixel_fraction_in_edge_band": fraction,
                 "changed_count_weighted_uniform_spatial_baseline": baseline,
                 "observed_minus_uniform_spatial_baseline": fraction - baseline if fraction is not None else None,
                 "ratio_to_uniform_spatial_baseline": fraction / baseline if baseline else None},
        "pair_transforms": comparison_summary(pairs),
        "source_to_capture_transforms": comparison_summary(source_comparisons),
        "before_first_vs_first": comparison_summary(before_first) if mode == "no_reset" else None,
        "cases": case_rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="t00_restore_diagnostic_summary")
    for mode in MODES:
        parser.add_argument("--" + mode.replace("_", "-") + "-result", default=f"results/t00_restore_{mode}.json")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.name):
        parser.error("Invalid result name")
    output = ROOT / "results" / f"{args.name}.json"
    metrics, sources, audits, errors = {"aggregation_status": "failed", "modes": {}}, [], [], []
    config = {"name": args.name, "requested_sources": vars(args), "source_provenance_checks": audits,
              "notes": ["Aggregation success does not imply RGB equality or diagnostic completion; source failures remain visible.",
                        "Only recorded observations enter min/median/max; missing values stay null and observation counts are explicit.",
                        "Histogram bins are absolute RGB channel differences 0..255, summed over available case histograms, including unchanged channels.",
                        "Edge baseline is the measured edge-band area divided by image area. The pooled baseline weights each case by its differing-pixel count, matching the pooled observed fraction.",
                        "The uniform spatial baseline is descriptive, not a significance test or a causal rendering/physics/shadow attribution.",
                        "Exact raw transform equality is retained; wrapped Euler deltas are descriptive, not quaternion geodesic errors.",
                        "All 20 configured case slots remain visible. Source-to-capture summaries have 40 expected comparisons.",
                        "Full-file SHA-256 covers source config as well as metrics; the unchanged provenance auditor is called before aggregation."]}
    try:
        spec = importlib.util.spec_from_file_location("restore_existing_auditor", ROOT / "scripts" / "audit_provenance.py")
        auditor = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(auditor)
        loaded = {}
        for mode in MODES:
            path = (ROOT / getattr(args, mode + "_result")).resolve()
            relative = path.relative_to(ROOT)
            if relative.parts[0] != "results" or path.suffix != ".json" or path == output:
                raise ValueError("Sources must be repository results JSONs distinct from the output")
            raw = path.read_bytes()
            data = json.loads(raw)
            validate_document(data, mode)
            audit = auditor.audit_file(str(path))
            audits.append(audit)
            if audit["fail"]:
                raise ValueError(f"{relative}: provenance failure: {'; '.join(audit['fail'])}")
            sources.append({"file": relative.as_posix(), "metrics_sha256": metrics_hash(data["metrics"]),
                            "sha256": hashlib.sha256(raw).hexdigest(), "exit_code": data["_meta"]["exit_code"],
                            "git_commit": data["_meta"].get("git_commit"), "mode": mode})
            loaded[mode] = data
        comparison_keys = ("scene", "seed", "resolution", "cases", "grid_size", "visibility_distance", "field_of_view", "quality", "edge_band_pixels")
        settings = [{key: data["config"].get(key) for key in comparison_keys} for data in loaded.values()]
        if any(setting != settings[0] for setting in settings[1:]):
            raise ValueError("Diagnostic modes differ in scene, seed, rendering, or edge-analysis settings")
        config["shared_source_settings"] = settings[0]
        metrics["modes"] = {mode: summarize_mode(data, mode) for mode, data in loaded.items()}
        metrics["aggregation_status"] = "passed"
    except Exception as error:
        errors.append(f"{type(error).__name__}: {error}")
    if errors:
        metrics["errors"] = errors
    payload = {"metrics": metrics, "config": config, "sources": sources}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print("FINAL_METRICS_JSON: " + json.dumps(metrics, sort_keys=True, allow_nan=False), flush=True)
    return 0 if metrics["aggregation_status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

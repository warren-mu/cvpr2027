"""Explicit D/E/F/G source summary; run through run_logged after commit.

Reuses the existing diagnostic statistics without changing a threshold. Full
per-case observations remain in each audited source; this output is compact.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase_progress import load_source
from restore_diagnostic_summary import summarize_mode
from src.envs.thor_common import write_result

MODES = ("no_reset_identity", "no_reset_edit_undo", "no_reset_revisit", "cross_process")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--source-prefix", required=True)
    args = parser.parse_args()
    if not all(re.fullmatch(r"[A-Za-z0-9_]+", value) for value in (args.name, args.source_prefix)):
        parser.error("Require explicit safe unique result names")
    output = ROOT / "results" / (args.name + ".json")
    if output.exists():
        raise FileExistsError("Preserve previous aggregate")
    metrics = {"aggregation_status": "failed", "modes": {}}
    sources, audits = [], []
    config = {"source_prefix": args.source_prefix, "source_provenance_checks": audits,
              "notes": ["All source cases are retained, including failures; exact-zero acceptance is unchanged.",
                        "Edge-band spatial comparison is descriptive, not a causal shadow/noise attribution.",
                        "Reported transform equality does not prove equality of all hidden Unity state.",
                        "No interpolation, correction of metadata, or fabricated cases."]}
    try:
        spec = importlib.util.spec_from_file_location("isolation_existing_audit", ROOT / "scripts/audit_provenance.py")
        auditor = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(auditor)
        settings = None
        for mode in MODES:
            data, ref, audit = load_source(f"results/{args.source_prefix}_{mode}.json", output, auditor)
            sources.append(ref)
            audits.append(audit)
            current = {key: data["config"].get(key) for key in
                       ("scene", "seed", "resolution", "cases", "grid_size", "visibility_distance", "field_of_view", "quality", "edge_band_pixels")}
            if settings is not None and current != settings:
                raise ValueError("D/E/F/G comparison configurations differ")
            settings = current
            summary = summarize_mode(data, mode)
            summary["cases"] = [{key: row.get(key) for key in ("state_index", "status", "error", "rgb", "edge")}
                                for row in summary["cases"]]
            summary["pbs_jobid"] = data["config"].get("pbs_jobid") or data["config"].get("runtime", {}).get("pbs_jobid")
            metrics["modes"][mode] = summary
        config["shared_source_settings"] = settings
        metrics["aggregation_status"] = "completed"
    except Exception as error:
        metrics["error"] = f"{type(error).__name__}: {error}"
    write_result(args.name, metrics, config)
    output.write_text(json.dumps({"metrics": metrics, "config": config, "sources": sources}, indent=2, allow_nan=False) + "\n")
    return 0 if metrics["aggregation_status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

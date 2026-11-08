"""Emit a provenance-logged, scheduler-charged T00b/T01 budget ledger."""
import argparse
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.envs.thor_common import write_result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("phase_submit", ROOT / "scripts/pbs/submit_phase.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    state = module.budget_state(ROOT)
    final = [r for r in state["submissions"] if r["basis"] == "final_accounting"]
    reserved = [r for r in state["submissions"] if r["basis"] != "final_accounting"]
    metrics = {"completed_scheduler_charged_gpu_hours": sum(r["accounted_or_reserved_gpu_hours"] for r in final),
               "unfinished_reserved_scheduler_charged_gpu_hours": sum(r["accounted_or_reserved_gpu_hours"] for r in reserved),
               "total_accounted_or_reserved_scheduler_charged_gpu_hours": state["accounted_or_reserved_total"],
               "limit_scheduler_charged_gpu_hours": module.TOTAL_LIMIT,
               "within_budget": state["accounted_or_reserved_total"] <= module.TOTAL_LIMIT}
    write_result(args.name, metrics, state)
    return 0 if metrics["within_budget"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

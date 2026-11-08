"""Record setup/diagnostic commands; run this through scripts/run_logged.py."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    split = sys.argv.index("--")
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    args = parser.parse_args(sys.argv[1:split])
    command = sys.argv[split + 1:]
    started = time.monotonic()
    error = None
    try:
        code = subprocess.call(command)
    except Exception as exc:
        code, error = 1, repr(exc)
    metrics = {"command_exit_code": code,
               "elapsed_seconds": time.monotonic() - started}
    result = {"metrics": metrics, "config": {"command": command,
              "pbs_job_id": os.environ.get("PBS_JOBID"),
              "scope": "environment setup or diagnostic, not a research result"},
              "error": error}
    path = Path(__file__).resolve().parents[2] / "results" / (args.name + ".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print("FINAL_METRICS_JSON: " + json.dumps(metrics), flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Submit one explicit T00 command to one Hopper GPU, with a saved immutable payload."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import uuid


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", required=True, help="exact run_logged identifier, starting t00_")
    ap.add_argument("--walltime", default="00:10:00")
    ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    ap.add_argument("--scratch", default=os.environ.get("CVPR_SCRATCH", f"/scratch/{os.environ.get('USER', '')}/cvpr2027"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--cpu-only", action="store_true", help="CPU setup probe only: t00_setup* names, 4 CPUs / 32 GB / no GPU")
    argv = sys.argv[1:]
    split = argv.index("--") if "--" in argv else len(argv)
    args = ap.parse_args(argv[:split])
    command = argv[split + 1:]
    if not command:
        ap.error("supply an explicit command after --")
    if not re.fullmatch(r"t00_[A-Za-z0-9_]+", args.name):
        ap.error("--name must start t00_ and contain only letters, numbers, and underscores")
    if args.cpu_only and not args.name.startswith("t00_setup"):
        ap.error("--cpu-only is restricted to t00_setup* runs")
    if not re.fullmatch(r"\d{2}:[0-5]\d:[0-5]\d", args.walltime):
        ap.error("--walltime must be HH:MM:SS")
    hours, minutes, seconds = map(int, args.walltime.split(":"))
    if not 0 < hours * 3600 + minutes * 60 + seconds <= 6 * 3600:
        ap.error("T00 requests must be positive and no longer than 06:00:00; track the cumulative six GPU-hour budget separately")
    repo = args.repo.expanduser().resolve()
    if not (repo / "scripts/run_logged.py").is_file():
        ap.error("--repo must be the shared project directory containing scripts/run_logged.py")
    if not Path(args.scratch).is_absolute():
        ap.error("--scratch must be absolute")
    for path in [repo / "results" / f"{args.name}.json",
                 repo / "logs/runs" / f"{args.name}.log",
                 repo / "logs/runs" / f"{args.name}.meta.json"]:
        if path.exists():
            ap.error(f"refusing to overwrite existing run artifacts; use a distinct name: {path}")
    token = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    directory = repo / "logs/pbs" / f"{args.name}-{token}"
    directory.mkdir(parents=True, exist_ok=False)
    payload_path = directory / "payload.json"
    environment = {"CVPR_SCRATCH": args.scratch}
    for key in ("CVPR_CUDA_MODULE", "CVPR_VULKAN_MODULE", "CVPR_RENDER_DEADLINE_EPOCH"):
        if key in os.environ:
            environment[key] = os.environ[key]
    payload = {"repo": str(repo), "name": args.name, "command": command, "environment": environment}
    with payload_path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    payload_path.chmod(0o444)
    template = (repo / "scripts/pbs/job_1gpu.pbs").read_text()
    if template.count("@@CVPR_PAYLOAD@@") != 1:
        ap.error("PBS template must contain exactly one payload placeholder")
    if args.cpu_only:
        template = template.replace("#PBS -l select=1:ngpus=1", "#PBS -l select=1:ncpus=4:mem=32gb:ngpus=0")
    script = directory / "job.pbs"
    script.write_text(template.replace("@@CVPR_PAYLOAD@@", shlex.quote(str(payload_path))))
    script.chmod(0o444)
    qsub = ["qsub", "-N", "cvpr27_" + token[-8:], "-l", f"walltime={args.walltime}",
            "-o", str(directory / "pbs.out"), str(script)]
    print(f"Payload: {payload_path}\nSubmit: {shlex.join(qsub)}", file=sys.stderr)
    if args.dry_run:
        print(script.read_text(), end="")
        return 0
    try:
        result = subprocess.run(qsub, capture_output=True, text=True, check=False)
    except OSError as error:
        (directory / "submission.json").write_text(json.dumps({"command": qsub, "cpu_only": args.cpu_only,
            "stdout": "", "stderr": str(error), "returncode": None}, indent=2) + "\n")
        print(f"Cannot execute qsub: {error}", file=sys.stderr)
        return 1
    (directory / "submission.json").write_text(json.dumps({"command": qsub, "cpu_only": args.cpu_only,
        "stdout": result.stdout, "stderr": result.stderr, "returncode": result.returncode}, indent=2) + "\n")
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())

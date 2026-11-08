"""Bounded read-only login-node checks for T00; invoke through run_logged."""
import json
from pathlib import Path
import subprocess


def main():
    checks = {
        "quota": ["hpc", "quota"],
        "legacy_workspace_policy": ["hpc", "workspace"],
        "modules": ["bash", "-lc", "export PAGER=cat MODULES_PAGER=cat; module -t avail"],
        "express_help": ["hpc", "autox"],
        "queues": ["qstat", "-Qf"],
    }
    for name, url in (("github", "https://github.com"),
                      ("pypi", "https://pypi.org/simple/"),
                      ("huggingface", "https://huggingface.co")):
        checks["network_" + name] = ["curl", "-L", "-s", "-o", "/dev/null", "--max-time", "10",
                                      "-w", "%{http_code} %{url_effective}\\n", url]
    observations = {}
    for name, command in checks.items():
        print("CHECK " + name + " " + json.dumps(command), flush=True)
        try:
            proc = subprocess.run(command, capture_output=True, text=True, timeout=90)
            observations[name] = {"returncode": proc.returncode,
                                  "stdout": proc.stdout, "stderr": proc.stderr}
        except subprocess.TimeoutExpired as exc:
            observations[name] = {"error": "timeout", "stdout": str(exc.stdout), "stderr": str(exc.stderr)}
        print(json.dumps(observations[name]), flush=True)
    root = Path(__file__).resolve().parents[2]
    path = root / "results" / "t00_login_preflight.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"metrics": {}, "config": {
        "scope": "login-node diagnostics", "observations": observations,
        "retention_caveat": "hpc workspace is legacy /hpctmp guidance; it does not establish current Hopper scratch retention."}}, indent=2) + "\n")
    print("FINAL_METRICS_JSON: {}", flush=True)


if __name__ == "__main__":
    main()

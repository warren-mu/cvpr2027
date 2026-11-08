"""Reserve a short, private Ray temp root on approved scratch; never start Ray.

Job 706439 failed before model initialization because Ray appended its session
and socket suffix to a long per-run RAY_TMPDIR. The installed Ray source uses
RAY_TMPDIR/ray/session_<date-with-microseconds>_<pid>/sockets/plasma_store
(_common/utils.py and _private/node.py), with a 107-byte Linux AF_UNIX limit.
Only this infrastructure path changes; every run/variant gets a fresh directory.
"""
import argparse
import datetime as dt
import importlib.metadata
import json
import os
from pathlib import Path
import re
import secrets

LINUX_SOCKET_LIMIT = 107
# Ten decimal PID digits cover even a signed 32-bit pid_t. A new exclusive
# directory has no existing socket, so Ray's initial filename has no .N suffix.
SOCKET_SUFFIX = "ray/session_9999-12-31_23-59-59_999999_2147483647/sockets/plasma_store"


def validate_socket_budget(directory):
    path = str(Path(directory) / SOCKET_SUFFIX)
    size = len(path.encode("utf-8"))
    if size > LINUX_SOCKET_LIMIT:
        raise ValueError(f"Ray scratch socket bound is {size} bytes, exceeds {LINUX_SOCKET_LIMIT}: {path}")
    return {"maximum_socket_path": path, "maximum_socket_path_bytes": size,
            "linux_socket_limit_bytes": LINUX_SOCKET_LIMIT, "pid_digits_reserved": 10}


def reserve_directory(scratch, name, pbs_jobid, work_dir, record_path, ray_version=None):
    """Atomically claim a fresh short name; never reuse/remove previous state."""
    scratch, work_dir, record_path = map(Path, (scratch, work_dir, record_path))
    if not scratch.is_absolute() or not work_dir.is_absolute() or not record_path.is_absolute():
        raise ValueError("Scratch, work directory and record path must be absolute")
    scratch, work_dir, record_path = scratch.resolve(), work_dir.resolve(), record_path.resolve()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or not pbs_jobid:
        raise ValueError("Require a safe run name and PBS job ID")
    if not work_dir.is_relative_to(scratch):
        raise ValueError("Raw work directory must remain under CVPR_SCRATCH")
    if record_path.exists():
        raise FileExistsError("Existing Ray temp provenance is preserved")
    parent = scratch / "r"
    if not parent.resolve().is_relative_to(scratch):
        raise ValueError("Ray scratch parent resolves outside CVPR_SCRATCH")
    # Check before creating any state. Long custom scratch roots fail explicitly;
    # they never fall back to home, /tmp, or another allocation's Ray cluster.
    validate_socket_budget(parent / "000000")
    parent.mkdir(parents=True, exist_ok=True)
    for _ in range(32):
        directory = parent / secrets.token_hex(3)
        try:
            directory.mkdir(mode=0o700)
            break
        except FileExistsError:
            continue
    else:
        raise FileExistsError("Could not reserve a fresh Ray temp directory")
    record = {"name": name, "pbs_jobid": pbs_jobid, "ray_tmpdir": str(directory),
              "ray_default_temp_dir": str(directory / "ray"), "raw_work_directory": str(work_dir),
              "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "ray_version": ray_version, "exclusive_directory": True,
              "retention": "preserve directory and Ray logs; no automatic deletion",
              "path_contract": SOCKET_SUFFIX, **validate_socket_budget(directory)}
    record_path.parent.mkdir(parents=True, exist_ok=True)
    # Persist inside the reservation as well as the small repository manifest.
    encoded = json.dumps(record, indent=2) + "\n"
    (directory / "reservation.json").write_text(encoded)
    with record_path.open("x") as stream:
        stream.write(encoded)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--record", required=True, type=Path)
    args = parser.parse_args()
    record = reserve_directory(os.environ["CVPR_SCRATCH"], args.name, os.environ["PBS_JOBID"],
                               args.work_dir, args.record, importlib.metadata.version("ray"))
    # stdout is consumed by the wrapper; do not import Ray or start any service.
    print(record["ray_tmpdir"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

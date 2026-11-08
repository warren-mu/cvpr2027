"""Frozen Ray worker startup hook: isolate compiler caches before actor tasks.

Standard library only. No torch/triton import, compilation-mode change, cache
deletion, GPU-mask change, or per-call environment mutation. Ordinary compiler
children inherit the worker's two cache paths, as required by TritonFuture.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import socket


CACHE_KEYS = ("TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR")
CONFIG_NAME = "t00b_worker_cache_config.json"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def validated_paths(config):
    paths = [Path(config[key]) for key in ("scratch", "working_directory", "worker_base")]
    if any(not p.is_absolute() or p.resolve() != p for p in paths):
        raise ValueError("Worker cache paths must be canonical absolute paths")
    scratch, work, base = paths
    if scratch == Path("/") or scratch == Path.home().resolve() or Path.home().resolve() in scratch.parents:
        raise ValueError("Worker caches require approved scratch outside HOME")
    if not work.is_relative_to(scratch) or work == scratch or base != work / "compile_cache/workers":
        raise ValueError("Worker cache path escapes this run's scratch directory")
    if not re.fullmatch(r"[0-9a-f]{64}", config["hook_source_sha256"]):
        raise ValueError("Missing frozen hook source hash")
    return scratch, work, base


def configure_worker_cache(config, inherited, *, pid, hostname, start_ticks, config_sha256):
    """Create one exclusive manifest and return an environment changing only two keys.

    A repeated call for the same process is idempotent only when the retained
    manifest, identity, and current environment agree. Incomplete/colliding
    directories are preserved and rejected rather than repaired or overwritten.
    """
    scratch, work, base = validated_paths(config)
    if inherited.get("PBS_JOBID") != config["pbs_jobid"] or Path(inherited["CVPR_SCRATCH"]).resolve() != scratch:
        raise ValueError("Worker cache PBS/scratch identity differs from the frozen run")
    if type(pid) is not int or pid <= 0 or not re.fullmatch(r"[A-Za-z0-9_.-]+", hostname):
        raise ValueError("Invalid worker process identity")
    if not re.fullmatch(r"[0-9]+", str(start_ticks)) or not re.fullmatch(r"[0-9a-f]{64}", config_sha256):
        raise ValueError("Missing process start time or frozen config hash")
    before = {key: inherited.get(key) for key in CACHE_KEYS}
    for value in before.values():
        if not value or not Path(value).is_absolute() or not Path(value).resolve().is_relative_to(work / "compile_cache"):
            raise ValueError("Worker did not inherit this run's compiler cache")
    directory = base / hostname / f"pid-{pid}-start-{start_ticks}"
    if directory.resolve() != directory:
        raise ValueError("Worker cache directory traverses a symlink")
    after = dict(zip(CACHE_KEYS, (str(directory / "triton"), str(directory / "inductor"))))
    identity = {"pid": pid, "host": hostname, "process_start_ticks": str(start_ticks),
                "pbs_jobid": config["pbs_jobid"], "name": config["name"],
                "working_directory": str(work), "worker_base": str(base),
                "worker_directory": str(directory), "environment": after,
                "hook_source_sha256": config["hook_source_sha256"], "hook_config_sha256": config_sha256}
    manifest = directory / "manifest.json"
    if manifest.exists():
        if manifest.is_symlink():
            raise ValueError("Worker cache manifest must not be a symlink")
        record = json.loads(manifest.read_text())
        if any(record.get(key) != value for key, value in identity.items()):
            raise ValueError("Existing worker manifest differs; preserving it")
        if before not in (record.get("previous_environment"), after):
            raise ValueError("Worker compiler environment changed after initial setup")
        if any(not Path(value).is_dir() or Path(value).resolve() != Path(value) for value in after.values()):
            raise ValueError("Existing worker cache directories are missing or redirected")
    else:
        directory.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        directory.mkdir(mode=0o700, exist_ok=False)
        for value in after.values():
            Path(value).mkdir(mode=0o700)
        record = {**identity, "previous_environment": before,
                  "scope": "one_Ray_worker_and_its_compiler_children",
                  "compiler_modes_changed": False, "manifest_file": str(manifest)}
        # O_EXCL preserves any previous complete or partial evidence.
        with os.fdopen(os.open(manifest, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as stream:
            stream.write(json.dumps(record, sort_keys=True, indent=2) + "\n")
    child = dict(inherited)
    child.update(after)
    return child, record


def setup_worker_cache():
    """Ray 2.60 calls this before worker.main_loop (before tasks/actors run)."""
    source = Path(__file__).resolve()
    raw = source.with_name(CONFIG_NAME).read_bytes()
    config = json.loads(raw)
    if digest(source.read_bytes()) != config["hook_source_sha256"]:
        raise ValueError("Frozen worker cache hook source changed")
    # /proc/self/stat field 22; split after ')' because comm may contain spaces.
    start_ticks = Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19]
    child, record = configure_worker_cache(config, os.environ, pid=os.getpid(),
                                           hostname=socket.gethostname(), start_ticks=start_ticks,
                                           config_sha256=digest(raw))
    for key in CACHE_KEYS:
        os.environ[key] = child[key]
    return record

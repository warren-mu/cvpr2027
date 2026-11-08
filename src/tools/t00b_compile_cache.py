"""Isolate generated compiler caches for one T00b run/variant, on scratch.

706436 failed loading a shared Triton __triton_launcher.so with ENOENT; the file
existed at later inspection. Isolation avoids reusing that shared state without
claiming a proven filesystem/race cause or disabling compilation. Installed
Triton FileCacheManager reads TRITON_CACHE_DIR; Torch 2.6 cache_dir_utils reads
TORCHINDUCTOR_CACHE_DIR. No package import, compilation, or cache deletion here.
"""
import hashlib
import json
from pathlib import Path


def freeze_worker_hook(work, overlay, log_dir, scratch, pbs_jobid, name):
    """Freeze the plain-Python hook and config into this run's private overlay."""
    source = Path(__file__).with_name("t00b_worker_cache.py")
    body = source.read_bytes()
    config = {"name": name, "pbs_jobid": pbs_jobid, "scratch": str(Path(scratch).resolve()),
              "working_directory": str(work), "worker_base": str(work / "compile_cache/workers"),
              "hook_source_sha256": hashlib.sha256(body).hexdigest()}
    raw = (json.dumps(config, sort_keys=True, indent=2) + "\n").encode()
    for directory in (overlay, log_dir):
        for filename, data in ((source.name, body), ("t00b_worker_cache_config.json", raw)):
            with (directory / filename).open("xb") as stream:
                stream.write(data)
    return {**config, "setup_hook": "t00b_worker_cache.setup_worker_cache",
            "hook_config_sha256": hashlib.sha256(raw).hexdigest(),
            "hook_file": str(overlay / source.name),
            "hook_config_file": str(overlay / "t00b_worker_cache_config.json")}


def validate_frozen_worker_hook(manifest):
    """Check frozen source/config before launching any training process."""
    config = manifest.get("worker_compilation_cache")
    if config is None:
        return
    work = Path(manifest["working_directory"]).resolve()
    overlay = work / "overlay"
    expected = {"hook_file": overlay / "t00b_worker_cache.py",
                "hook_config_file": overlay / "t00b_worker_cache_config.json"}
    for key, path in expected.items():
        if config.get(key) != str(path) or path.resolve() != path:
            raise ValueError("Frozen worker hook must be a private overlay file")
    if config.get("setup_hook") != "t00b_worker_cache.setup_worker_cache":
        raise ValueError("Unexpected worker cache startup hook")
    for key in ("name", "pbs_jobid", "working_directory"):
        if config.get(key) != manifest[key]:
            raise ValueError(f"Frozen worker hook identity differs: {key}")
    source = expected["hook_file"].read_bytes()
    raw = expected["hook_config_file"].read_bytes()
    if (hashlib.sha256(source).hexdigest() != config["hook_source_sha256"]
            or hashlib.sha256(raw).hexdigest() != config["hook_config_sha256"]):
        raise ValueError("Frozen worker hook/config hash changed")
    payload = json.loads(raw)
    if payload != {key: config[key] for key in
                   ("name", "pbs_jobid", "scratch", "working_directory", "worker_base", "hook_source_sha256")}:
        raise ValueError("Frozen hook config differs from run manifest")
    from src.tools.t00b_worker_cache import validated_paths
    validated_paths(payload)


def worker_manifest_evidence(manifest):
    """Return exact worker JSON and hashes for the existing archived resource log."""
    config = manifest.get("worker_compilation_cache")
    if config is None:
        return []  # Read compatibility for previously frozen run manifests.
    work = Path(manifest["working_directory"]).resolve()
    base = Path(config["worker_base"])
    if base != work / "compile_cache/workers" or base.resolve() != base:
        raise ValueError("Worker evidence root differs from this run")
    paths = sorted(base.glob("*/pid-*/manifest.json"))
    if len(paths) > 4096:
        raise ValueError("Unexpected worker manifest volume; scratch originals retained")
    rows = []
    for path in paths:
        if path.resolve() != path or path.stat().st_size > 65536:
            raise ValueError("Worker manifest redirected or exceeds evidence bound")
        raw = path.read_bytes()
        record = json.loads(raw)
        for key in ("pbs_jobid", "name", "working_directory", "worker_base", "hook_source_sha256", "hook_config_sha256"):
            if record.get(key) != config[key]:
                raise ValueError(f"Worker manifest disagrees with frozen config: {key}")
        if record.get("manifest_file") != str(path) or record.get("worker_directory") != str(path.parent):
            raise ValueError("Worker evidence has inconsistent paths")
        rows.append({"type": "worker_compilation_cache", "manifest_file": str(path),
                     "manifest_sha256": hashlib.sha256(raw).hexdigest(),
                     "manifest_raw_json": raw.decode("utf-8")})
    return rows


def cache_environment(work_dir):
    work = Path(work_dir)
    if not work.is_absolute():
        raise ValueError("Compiler cache work directory must be absolute")
    root = work.resolve() / "compile_cache"
    return {"TRITON_CACHE_DIR": str(root / "triton"),
            "TORCHINDUCTOR_CACHE_DIR": str(root / "inductor")}


def prepare_environment(manifest, inherited):
    """Return a child environment changing only two generated-cache paths."""
    work = Path(manifest["working_directory"]).resolve()
    scratch = Path(inherited["CVPR_SCRATCH"]).resolve()
    if not work.is_relative_to(scratch) or manifest.get("pbs_jobid") != inherited.get("PBS_JOBID"):
        raise ValueError("Compiler cache requires this PBS run's approved scratch directory")
    values = cache_environment(work)
    if manifest.get("compilation_cache_environment") != values:
        raise ValueError("Compiler cache paths do not match the recorded run manifest")
    validate_frozen_worker_hook(manifest)
    root = work / "compile_cache"
    # Preserve all previous caches, including partially initialized runs.
    root.mkdir(mode=0o700, exist_ok=False)
    for path in values.values():
        Path(path).mkdir(mode=0o700)
    child = dict(inherited)
    child.update(values)
    scope = ("driver cache for this run; Ray workers use frozen startup hook private caches"
             if manifest.get("worker_compilation_cache")
             else "one_run_and_variant; ranks within this run share its cache")
    record = {"scope": scope,
              "pbs_jobid": manifest["pbs_jobid"], "name": manifest["name"],
              "environment": values,
              "previous_environment": {key: inherited.get(key) for key in values},
              "retention": "all old and new caches preserved on scratch",
              "compiler_modes_changed": False}
    return child, record

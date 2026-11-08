"""Isolate generated compiler caches for one T00b run/variant, on scratch.

706436 failed loading a shared Triton __triton_launcher.so with ENOENT; the file
existed at later inspection. Isolation avoids reusing that shared state without
claiming a proven filesystem/race cause or disabling compilation. Installed
Triton FileCacheManager reads TRITON_CACHE_DIR; Torch 2.6 cache_dir_utils reads
TORCHINDUCTOR_CACHE_DIR. No package import, compilation, or cache deletion here.
"""
from pathlib import Path


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
    root = work / "compile_cache"
    # Preserve all previous caches, including partially initialized runs.
    root.mkdir(mode=0o700, exist_ok=False)
    for path in values.values():
        Path(path).mkdir(mode=0o700)
    child = dict(inherited)
    child.update(values)
    record = {"scope": "one_run_and_variant; ranks within this run share its cache",
              "pbs_jobid": manifest["pbs_jobid"], "name": manifest["name"],
              "environment": values,
              "previous_environment": {key: inherited.get(key) for key in values},
              "retention": "all old and new caches preserved on scratch",
              "compiler_modes_changed": False}
    return child, record

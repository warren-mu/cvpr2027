"""Reproducible vLLM 0.8.2 NVML UUID compatibility repair; run only in PBS.

PBS 706199 failed because upstream converted CUDA_VISIBLE_DEVICES UUID to int.
This only translates that same UUID into NVML's physical index. It never
rewrites CUDA_VISIBLE_DEVICES, selects a different GPU, or changes model code.
Upstream: https://github.com/vllm-project/vllm/blob/v0.8.2/vllm/platforms/cuda.py
"""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path


ORIGINAL_SHA256 = "3315afae41183ff0628982fdb08ef3997ab1694a2841a25381cdb2f70b31dae6"
OLD = """        physical_device_id = device_ids[device_id]
        return int(physical_device_id)
"""
NEW = """        physical_device_id = device_ids[device_id]
        # CVPR2027: retain PBS UUID visibility; resolve only its NVML index.
        if physical_device_id.startswith("GPU-"):
            from uuid import UUID
            if "GPU-" + str(UUID(physical_device_id[4:])) != physical_device_id:
                raise ValueError("Expected a complete canonical PBS GPU UUID")
            pynvml.nvmlInit()
            try:
                handle = pynvml.nvmlDeviceGetHandleByUUID(physical_device_id)
                actual = pynvml.nvmlDeviceGetUUID(handle)
                if isinstance(actual, bytes):
                    actual = actual.decode("ascii")
                if actual != physical_device_id:
                    raise RuntimeError("NVML UUID identity mismatch")
                return int(pynvml.nvmlDeviceGetIndex(handle))
            finally:
                pynvml.nvmlShutdown()
        return int(physical_device_id)
"""


def sha(data):
    return hashlib.sha256(data).hexdigest()


def repaired_bytes(original):
    if sha(original) != ORIGINAL_SHA256:
        raise RuntimeError("Unrecognized vLLM source; refusing an unreviewed patch")
    source = original.decode("utf-8")
    if source.count(OLD) != 1:
        raise RuntimeError("Expected exactly one upstream device mapping block")
    return source.replace(OLD, NEW).encode("utf-8")


def main():
    if not os.environ.get("PBS_JOBID"):
        raise RuntimeError("Apply dependency repair only in a PBS allocation")
    dist = importlib.metadata.distribution("vllm")
    if dist.version != "0.8.2":
        raise RuntimeError("This repair is restricted to exact vLLM 0.8.2")
    target = Path(dist.locate_file("vllm/platforms/cuda.py")).resolve()
    prefix = Path(os.environ["CVPR_CONDA_PREFIX"]).resolve()
    scratch = Path(os.environ["CVPR_SCRATCH"]).resolve()
    if prefix not in target.parents or scratch not in prefix.parents:
        raise RuntimeError("Refusing to patch outside this project's scratch environment")
    backup = target.with_name("cuda.py.cvpr2027-original-" + ORIGINAL_SHA256)
    current = target.read_bytes()
    if sha(current) == ORIGINAL_SHA256:
        if backup.exists():
            if backup.read_bytes() != current:
                raise RuntimeError("Existing original backup differs; preserved")
        else:
            with backup.open("xb") as handle:
                handle.write(current)
        patched = repaired_bytes(current)
        target.write_bytes(patched)
        action = "applied"
    else:
        if not backup.is_file():
            raise RuntimeError("Unknown source and no verified original backup")
        patched = repaired_bytes(backup.read_bytes())
        if current != patched:
            raise RuntimeError("Existing source is neither original nor this exact repair")
        action = "verified_existing"
    print("VLLM_UUID_REPAIR: " + json.dumps({
        "action": action, "path": str(target), "backup": str(backup),
        "original_sha256": ORIGINAL_SHA256, "patched_sha256": sha(patched),
        "cuda_visible_devices_unchanged": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "scope": "NVML UUID-to-index discovery only; model and training config unchanged",
    }), flush=True)


if __name__ == "__main__":
    main()

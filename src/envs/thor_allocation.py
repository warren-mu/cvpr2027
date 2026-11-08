"""PBS GPU identities and private AI2-THOR caches for T00b's two-GPU job.

Importing this module does not initialize a GPU. Call prepare_allocation only
inside PBS. Neither helper changes HOME or CUDA_VISIBLE_DEVICES.
"""
import csv
import ctypes
import importlib.metadata
import json
import numbers
import os
from pathlib import Path
import re
import subprocess
import uuid


def _command(argv, timeout=45):
    completed = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    if completed.returncode:
        raise RuntimeError(f"{argv!r}: exit {completed.returncode}: {completed.stderr[-3000:]}")
    return completed.stdout


def match_devices(mask, rows, expected_gpus):
    """Resolve the ordered PBS mask without assuming CUDA and NVML ordinals agree."""
    tokens = [value.strip() for value in mask.split(",")]
    if len(tokens) != expected_gpus or any(not value for value in tokens):
        raise RuntimeError("PBS GPU mask does not match the requested GPU count")
    numeric = all(value.isdigit() for value in tokens)
    full_uuid = all(re.fullmatch(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", value)
                    for value in tokens)
    if not (numeric or full_uuid):
        raise RuntimeError("Require a homogeneous numeric mask or complete non-MIG GPU UUIDs")
    devices = []
    for logical, token in enumerate(tokens):
        candidates = [row for row in rows if row[0] == token] if numeric else [
            row for row in rows if row[1].lower() == token.lower()]
        if len(candidates) != 1:
            raise RuntimeError(f"PBS GPU identity is not unique: {token!r}")
        index, identity, name = candidates[0]
        if "H200" not in name:
            raise RuntimeError(f"Expected H200, found {name!r}")
        devices.append({"logical_index": logical, "gpu_index": int(index),
                        "gpu_uuid": identity, "gpu_name": name})
    if len({item["gpu_uuid"] for item in devices}) != expected_gpus:
        raise RuntimeError("PBS mask assigns the same physical GPU more than once")
    return devices


def cuda_driver_uuids():
    cuda = ctypes.CDLL("libcuda.so.1")

    def checked(name, *args):
        code = getattr(cuda, name)(*args)
        if code:
            raise RuntimeError(f"{name} returned CUDA error {code}")

    checked("cuInit", 0)
    count = ctypes.c_int()
    checked("cuDeviceGetCount", ctypes.byref(count))
    identities = []
    for ordinal in range(count.value):
        device = ctypes.c_int()
        checked("cuDeviceGet", ctypes.byref(device), ordinal)
        raw = (ctypes.c_ubyte * 16)()
        checked("cuDeviceGetUuid_v2", ctypes.byref(raw), device)
        identities.append("GPU-" + str(uuid.UUID(bytes=bytes(raw))))
    return identities


def prepare_allocation(name, expected_gpus=2):
    if (not re.fullmatch(r"[A-Za-z0-9_]+", name) or type(expected_gpus) is not int
            or not 1 <= expected_gpus <= 8):
        raise ValueError("Invalid allocation name or GPU count")
    if not os.environ.get("PBS_JOBID"):
        raise RuntimeError("GPU validation requires a PBS allocation")
    mask = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    scratch_text = os.environ.get("CVPR_SCRATCH", "")
    if not scratch_text or not Path(scratch_text).is_absolute():
        raise RuntimeError("CVPR_SCRATCH must name an absolute scratch directory")
    scratch = Path(scratch_text).resolve()
    repo = Path(__file__).resolve().parents[2]
    log_dir = repo / "logs" / "runs" / name
    log_dir.mkdir(parents=True, exist_ok=True)
    raw = _command(["nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader"])
    rows = [[value.strip() for value in row] for row in csv.reader(raw.splitlines()) if row]
    devices = match_devices(mask, rows, expected_gpus)
    actual_cuda = cuda_driver_uuids()
    if actual_cuda != [item["gpu_uuid"] for item in devices]:
        raise RuntimeError("CUDA driver logical device UUIDs differ from the PBS/NVML order")
    vk_text = _command(["vulkaninfo"])
    (log_dir / "vulkaninfo_full.txt").write_text(vk_text)
    vk_map, current = {}, None
    for line in vk_text.splitlines():
        header = re.match(r"\s*GPU(\d+):", line)
        if header:
            current = int(header.group(1))
        identity = re.search(r"deviceUUID\s*=\s*([0-9a-fA-F-]+)", line)
        if identity and current is not None:
            key = identity.group(1).lower()
            if key in vk_map and vk_map[key] != current:
                raise RuntimeError("Vulkan UUID has inconsistent ordinals")
            vk_map[key] = current
    for item in devices:
        mig = _command(["nvidia-smi", "-i", item["gpu_uuid"],
                        "--query-gpu=mig.mode.current", "--format=csv,noheader"]).strip()
        if mig.lower() != "disabled":
            raise RuntimeError(f"Require whole GPUs with MIG disabled, found {mig!r}")
        identity = item["gpu_uuid"].removeprefix("GPU-").lower()
        if identity not in vk_map:
            raise RuntimeError("Allocated GPU is absent from Vulkan; CPU fallback is not accepted")
        item.update(vulkan_device_index=vk_map[identity], mig_mode=mig)
    version = importlib.metadata.version("ai2thor")
    if version != "5.0.0":
        raise RuntimeError(f"Require ai2thor==5.0.0, found {version}")
    result = {"name": name, "pbs_jobid": os.environ["PBS_JOBID"],
              "cuda_visible_devices": mask, "devices": devices,
              "cuda_driver_uuids": actual_cuda, "scratch": str(scratch),
              "log_dir": str(log_dir), "ai2thor_version": version,
              "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None}
    print("ALLOCATION_CONFIG: " + json.dumps(result, sort_keys=True), flush=True)
    return result


def make_controller_class(runtime, role="navigation"):
    """Return an upstream Controller subclass using only the recorded PBS GPUs.

    gpu_device is a logical slot in the original PBS mask. Each instance owns
    its mapping, temporary files, and Unity log; immutable releases are shared.
    """
    import ai2thor.build
    from ai2thor.controller import Controller

    if not re.fullmatch(r"[A-Za-z0-9_-]+", role):
        raise ValueError("Invalid controller role")
    ai2thor.build.base_url = ai2thor.build.base_url.replace("http://", "https://", 1)
    devices = runtime["devices"]
    mapping = {str(item["gpu_index"]): item["vulkan_device_index"] for item in devices}
    base = Path(runtime["scratch"]) / "ai2thor"

    class AllocationController(Controller):
        def __init__(self, *args, **kwargs):
            if os.environ.get("PBS_JOBID") != runtime["pbs_jobid"]:
                raise RuntimeError("Controller runtime belongs to another PBS job")
            mask = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            if mask != runtime["cuda_visible_devices"]:
                raise RuntimeError("Server CUDA mask changed after allocation validation")
            logical = kwargs.get("gpu_device", 0)
            if not isinstance(logical, numbers.Integral) or not 0 <= logical < len(devices):
                raise ValueError("Controller gpu_device must be a validated PBS logical slot")
            item = devices[int(logical)]
            # AI2-THOR remaps numeric masks itself, but ignores UUID masks.
            kwargs["gpu_device"] = int(logical) if all(x.strip().isdigit() for x in mask.split(",")) else item["gpu_index"]
            token = f"{os.getpid()}-{uuid.uuid4().hex}"
            self._cvpr_session = Path(runtime["scratch"]) / "thor_sessions" / runtime["pbs_jobid"] / role / token
            self._cvpr_log = Path(runtime["log_dir"]) / "unity" / role / token
            self._cvpr_session.mkdir(parents=True, exist_ok=False)
            self._cvpr_log.mkdir(parents=True, exist_ok=False)
            (self._cvpr_session / "cuda-vulkan-mapping.json").write_text(json.dumps(mapping) + "\n")
            self.cvpr_gpu_identity = dict(item)
            super().__init__(*args, **kwargs)

        @property
        def base_dir(self):
            return str(self._cvpr_session)

        @property
        def releases_dir(self):
            return str(base / "releases")

        @property
        def log_dir(self):
            return str(self._cvpr_log)

        def unity_command(self, width, height, headless):
            return super().unity_command(width, height, headless) + ["-logFile", str(self._cvpr_log / "Player.log")]

        def prune_releases(self):
            return None

    return AllocationController

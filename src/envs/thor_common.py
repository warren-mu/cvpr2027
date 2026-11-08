"""Bounded, allocation-aware helpers for the T00 rendering programs.

No AI2-THOR, CUDA, or NVML initialization occurs on import. Run measurements
through scripts/run_logged.py, inside a PBS allocation.
"""
from contextlib import contextmanager
import csv
import importlib.metadata
import json
import multiprocessing as mp
from multiprocessing.connection import wait
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]


def command(argv, timeout=30):
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    if proc.returncode:
        raise RuntimeError(f"{argv!r}: exit {proc.returncode}: {proc.stderr[-4000:]}")
    return proc.stdout


def interrupted(signum, _frame):
    if signum == signal.SIGALRM:
        raise TimeoutError("Worker deadline expired")
    raise KeyboardInterrupt(f"Interrupted by signal {signum}")


def install_signals():
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM):
        signal.signal(sig, interrupted)


def deadline_after(seconds):
    deadline = time.time() + seconds
    if os.environ.get("CVPR_RENDER_DEADLINE_EPOCH"):
        # Reserve time for cleanup/result serialization before the job deadline.
        deadline = min(deadline, float(os.environ["CVPR_RENDER_DEADLINE_EPOCH"]) - 30)
    if deadline <= time.time():
        raise TimeoutError("The shared rendering budget has expired")
    return deadline


def write_result(name, metrics, config):
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise ValueError("Invalid run name")
    path = ROOT / "results" / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"metrics": metrics, "config": config}
    # allow_nan=False prevents invalid or invented non-finite measurement values.
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print("FINAL_METRICS_JSON: " + json.dumps(metrics, sort_keys=True, allow_nan=False), flush=True)


def prepare_runtime(name):
    """Validate the scheduler's GPU and Vulkan identity without changing masks."""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise ValueError("Invalid run name")
    if not os.environ.get("PBS_JOBID"):
        raise RuntimeError("GPU measurements require a PBS allocation")
    mask = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not mask or "," in mask or mask.startswith("MIG-"):
        raise RuntimeError("Expected exactly one whole PBS-assigned GPU in CUDA_VISIBLE_DEVICES")
    scratch_value = os.environ.get("CVPR_SCRATCH")
    if not scratch_value:
        raise RuntimeError("Set CVPR_SCRATCH to the allocated scratch workspace")
    scratch = Path(scratch_value).expanduser().resolve()
    scratch.mkdir(parents=True, exist_ok=True)
    log_dir = ROOT / "logs" / "runs" / name
    log_dir.mkdir(parents=True, exist_ok=True)
    gpu_text = command(["nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader"])
    rows = [[x.strip() for x in row] for row in csv.reader(gpu_text.splitlines()) if row]
    matches = [row for row in rows if (row[1].startswith(mask) if mask.startswith("GPU-") else row[0] == mask)]
    if len(matches) != 1:
        raise RuntimeError(f"Cannot uniquely verify PBS GPU mask {mask!r} against nvidia-smi")
    index, uuid, gpu_name = matches[0]
    if "H200" not in gpu_name:
        raise RuntimeError(f"T00 requires an H200; allocated device is {gpu_name}")
    mig = command(["nvidia-smi", "-i", uuid, "--query-gpu=mig.mode.current", "--format=csv,noheader"]).strip()
    if mig.lower() != "disabled":
        raise RuntimeError(f"Whole-GPU graphics requires MIG disabled; reported {mig!r}")
    vk = command(["vulkaninfo"], timeout=45)
    (log_dir / "vulkaninfo_full.txt").write_text(vk)
    # Match UUID, not ordinal: CUDA and Vulkan enumerate devices differently.
    current = None
    vk_map = {}
    for line in vk.splitlines():
        match = re.match(r"\s*GPU(\d+):", line)
        if match:
            current = int(match.group(1))
        match = re.search(r"deviceUUID\s*=\s*([0-9a-fA-F-]+)", line)
        if match and current is not None:
            vk_map[match.group(1).lower()] = current
    bare_uuid = uuid.removeprefix("GPU-").lower()
    if bare_uuid not in vk_map:
        raise RuntimeError("Vulkan does not enumerate the allocated NVIDIA UUID; CPU fallback is not accepted")
    version = importlib.metadata.version("ai2thor")
    if version != "5.0.0":
        raise RuntimeError(f"Pinned VAGEN requires ai2thor==5.0.0, found {version}")
    runtime = {
        "pbs_jobid": os.environ["PBS_JOBID"], "cuda_visible_devices": mask,
        "gpu_index": int(index), "gpu_uuid": uuid, "gpu_name": gpu_name,
        "vulkan_device_index": vk_map[bare_uuid], "mig_mode": mig,
        "ai2thor_version": version, "scratch": str(scratch), "log_dir": str(log_dir),
        # THOR 5 interprets numeric masks as logical ordinals; UUID masks are
        # ignored by its parser, so use the nvidia-smi index in that case.
        "controller_gpu_device": 0 if mask.isdigit() else int(index),
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
    }
    print("RUNTIME_CONFIG: " + json.dumps(runtime, sort_keys=True), flush=True)
    return runtime


@contextmanager
def controller_session(config, worker_id, segmentation=False, connection=None):
    """Keep build/cache files on scratch and isolate each mapping and Unity log."""
    import ai2thor.build
    from ai2thor.controller import Controller
    from ai2thor.platform import CloudRendering

    runtime = config["runtime"]
    base = Path(runtime["scratch"]) / "ai2thor"
    session = Path(runtime["scratch"]) / "t00_sessions" / config["name"] / str(os.getpid())
    session.mkdir(parents=True, exist_ok=True)
    log_dir = Path(runtime["log_dir"]) / worker_id
    log_dir.mkdir(parents=True, exist_ok=True)
    # Cache identities depend on node enumeration; never reuse a previous job's
    # cuda-vulkan-mapping.json. The shared release cache itself is reusable.
    (session / "cuda-vulkan-mapping.json").write_text(json.dumps({
        str(runtime["gpu_index"]): runtime["vulkan_device_index"]
    }))
    os.environ["TMPDIR"] = str(session)
    os.environ["XDG_CACHE_HOME"] = str(Path(runtime["scratch"]) / "cache")
    os.environ["XDG_CONFIG_HOME"] = str(session / "config")

    class ScratchController(Controller):
        @property
        def base_dir(self):
            return str(session)

        @property
        def releases_dir(self):
            return str(base / "releases")

        @property
        def log_dir(self):
            return str(log_dir)

        def unity_command(self, width, height, headless):
            args = super().unity_command(width, height, headless)
            return args + ["-logFile", str(log_dir / "Player.log")]

        def prune_releases(self):
            # Concurrent workers share immutable build downloads; do not let a
            # controller prune another worker's or earlier job's cached build.
            return None

    # The 5.0 public build endpoint uses HTTP; HTTPS identifies the same build
    # and retains upstream SHA-256 verification.
    ai2thor.build.base_url = ai2thor.build.base_url.replace("http://", "https://", 1)
    controller = ScratchController.__new__(ScratchController)
    try:
        controller.__init__(
            scene=config["scene"], platform=CloudRendering,
            gpu_device=runtime["controller_gpu_device"],
            width=config["resolution"], height=config["resolution"],
            agentMode="default", renderDepthImage=False,
            renderInstanceSegmentation=segmentation,
            gridSize=config.get("grid_size", 0.1),
            visibilityDistance=config.get("visibility_distance", 10),
            fieldOfView=config.get("field_of_view", 100),
            server_timeout=45, server_start_timeout=90,
        )
        if connection is not None:
            unity = process_identity(controller.unity_pid)
            if unity:
                connection.send(("unity_identity", unity))
        yield controller
    finally:
        original_error = sys.exc_info()[0] is not None
        signal.setitimer(signal.ITIMER_REAL, 15)
        try:
            controller.stop()
        except BaseException:
            traceback.print_exc()
            if not original_error:
                raise
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)


def valid_frame(event, resolution):
    import numpy as np
    frame = event.frame
    if frame is None or frame.shape != (resolution, resolution, 3) or not np.any(frame):
        raise RuntimeError(f"Invalid/empty rendered frame: {getattr(frame, 'shape', None)}")
    return frame


def step(controller, action, **kwargs):
    event = controller.step(action=action, **kwargs)
    if not event.metadata.get("lastActionSuccess", False):
        raise RuntimeError(f"{action}: {event.metadata.get('errorMessage', 'action failed')}")
    return event


def _worker_entry(fn, config, connection):
    os.setsid()  # One private group containing this worker and its Unity child.
    install_signals()
    signal.setitimer(signal.ITIMER_REAL, max(0.1, config["deadline"] - time.time()))
    try:
        result = fn(config, connection)
        connection.send(("result", result))
    except BaseException as error:
        traceback.print_exc()
        try:
            connection.send(("error", {"type": type(error).__name__, "message": str(error)}))
        except (OSError, EOFError):
            pass
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        # Keep the leader alive after publishing its result. The parent must
        # signal this private group before reaping its leader, so its PGID can
        # never be reused between the ownership check and group cleanup.
        try:
            if connection.poll(max(0.1, config["deadline"] + 30 - time.time())):
                connection.recv()
        except (OSError, EOFError, KeyboardInterrupt):
            pass
        finally:
            connection.close()


def process_identity(pid):
    """Linux identity: PID plus kernel start ticks, with group/session scope."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return {"pid": pid, "start_ticks": int(fields[19]),
                "pgid": int(fields[2]), "sid": int(fields[3])}
    except (FileNotFoundError, ProcessLookupError):
        return None


def signal_identity(identity, signum):
    """Signal a known surviving child without a PID-reuse race (Linux pidfd)."""
    if not identity:
        return
    pid = identity["pid"]
    try:
        fd = os.pidfd_open(pid)
    except ProcessLookupError:
        return
    try:
        current = process_identity(pid)
        if current and current["start_ticks"] == identity["start_ticks"]:
            signal.pidfd_send_signal(fd, signum)
    except ProcessLookupError:
        pass
    finally:
        os.close(fd)


def stop_workers(workers):
    """TERM/KILL all owned groups together, with a shared four-second bound."""
    errors = []
    # A second termination request must not interrupt scoped cleanup halfway.
    handlers = {sig: signal.signal(sig, signal.SIG_IGN)
                for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM)}
    stop_deadline = time.monotonic() + 4
    try:
        for signum in (signal.SIGTERM, signal.SIGKILL):
            for worker in workers:
                try:
                    identity = worker.identity
                    current = process_identity(worker.process.pid)
                    if (identity and current and
                            current["start_ticks"] == identity["start_ticks"] and
                            current["pgid"] == current["sid"] == worker.process.pid):
                        # No join/is_alive/exitcode calls until ALL group
                        # signals finish. Even an exited leader stays unreaped.
                        os.killpg(worker.process.pid, signum)
                    else:
                        # No verified private group: use pidfds only, never a
                        # saved PGID or a bare PID that could belong to a peer.
                        if signum == signal.SIGTERM:
                            errors.append(f"pid={worker.process.pid}: private group ownership unavailable; only verified pidfds may be signaled")
                        signal_identity(identity, signum)
                        signal_identity(worker.unity_identity, signum)
                except ProcessLookupError:
                    pass
                except Exception as error:
                    errors.append(f"pid={worker.process.pid}: {type(error).__name__}: {error}")
            if signum == signal.SIGTERM and workers:
                time.sleep(min(2, max(0, stop_deadline - time.monotonic())))
        for worker in workers:
            worker.process.join(timeout=max(0, stop_deadline - time.monotonic()))
            if worker.process.exitcode is None:
                errors.append(f"pid={worker.process.pid}: worker did not exit within cleanup deadline")
            worker.connection.close()
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    return errors


class Worker:
    def __init__(self, fn, config):
        context = mp.get_context("spawn")
        self.connection, child = context.Pipe()
        self.process = context.Process(target=_worker_entry, args=(fn, config, child))
        self.process.start()
        self.identity = process_identity(self.process.pid)
        self.unity_identity = None
        child.close()
        self.latest = {}
        self.result = None
        self.error = None
        self.ready = False

    def receive(self):
        while self.connection.poll():
            try:
                kind, value = self.connection.recv()
            except EOFError:
                break
            if kind == "progress":
                self.latest = value
            elif kind == "ready":
                self.ready = True
                self.latest.update(value)
            elif kind == "result":
                self.result = value
                self.latest = value
            elif kind == "error":
                self.error = value
            elif kind == "unity_identity":
                self.unity_identity = value
        if (wait([self.process.sentinel], timeout=0) and
                self.result is None and self.error is None):
            self.error = {"type": "WorkerExit", "message": "Worker exited without a result; not reaped until cleanup"}

    def stop(self):
        return stop_workers([self])

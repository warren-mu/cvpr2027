#!/usr/bin/env python3
"""Supervise T00b training, tee output, and sample both allocated GPUs plus owned process trees."""

import argparse
import csv
import json
import os
from pathlib import Path
import re
import selectors
import signal
import subprocess
import sys
import time
from src.tools.t00b_compile_cache import prepare_environment, worker_manifest_evidence


def main():
    split = sys.argv.index("--") if "--" in sys.argv else len(sys.argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-log", required=True, type=Path)
    parser.add_argument("--resource-log", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--server-pid", required=True, type=int)
    parser.add_argument("--server-create-time", required=True, type=float)
    parser.add_argument("--timeout", type=float, default=3600)
    args = parser.parse_args(sys.argv[1:split])
    command = sys.argv[split + 1:]
    if not command:
        parser.error("Missing command after --")
    import psutil
    manifest = json.loads(args.manifest.read_text())
    gpu_uuids = [device["gpu_uuid"] for device in manifest["allocation"]["devices"]]
    if len(gpu_uuids) != 2 or len(set(gpu_uuids)) != 2:
        raise ValueError("Expected exactly two verified allocation GPU UUIDs")

    args.train_log.parent.mkdir(parents=True, exist_ok=True)
    args.resource_log.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    process = None
    training_known, server_known, resource_known = {}, {}, {}
    previous_cpu = {}
    previous_sample = None
    interrupted = {"signal": None}
    timed_out = False
    returncode = 1

    def on_signal(signum, _frame):
        interrupted["signal"] = signum

    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, on_signal)

    def discover(root_pid, known, expected_birth=None):
        try:
            root = psutil.Process(root_pid)
            if expected_birth is not None and root.create_time() != expected_birth:
                raise psutil.NoSuchProcess(root_pid)
            candidates = [root, *root.children(recursive=True)]
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            candidates = []
        # Keep previously observed descendants even if Ray reparented them.
        for proc in list(known.values()):
            try:
                if proc.is_running():
                    candidates.extend(proc.children(recursive=True))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        for proc in candidates:
            try:
                known[(proc.pid, proc.create_time())] = proc
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

    with args.train_log.open("wb") as logfile, args.resource_log.open("w") as resources:
        def emit(row):
            resources.write(json.dumps(row, allow_nan=False) + "\n")
            resources.flush()

        try:
            training_environment, cache_record = prepare_environment(manifest, os.environ)
            emit({"type": "compilation_cache", **cache_record})
            print("T00B_COMPILATION_CACHE: " + json.dumps(cache_record, sort_keys=True), flush=True)
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       start_new_session=True, bufsize=0, env=training_environment)
            selector = selectors.DefaultSelector()
            selector.register(process.stdout, selectors.EVENT_READ)
            next_sample = started
            eof = False
            exit_seen = None
            console_buffer = ""
            while True:
                now = time.monotonic()
                if (Path(manifest["working_directory"]) / "rgb_failure.json").exists():
                    print("T00B_INVALID_RGB: stopping training; raw failure event retained", flush=True)
                    returncode = 1
                    break
                if interrupted["signal"] is not None or now - started >= args.timeout:
                    timed_out = interrupted["signal"] is None
                    returncode = 124 if timed_out else 128 + interrupted["signal"]
                    break
                if now >= next_sample:
                    discover(process.pid, training_known)
                    resource_known.update(training_known)
                    discover(args.server_pid, server_known, args.server_create_time)
                    resource_known.update(server_known)
                    rss, cpu_delta, count = 0, 0.0, 0
                    current_cpu = {}
                    for key, proc in list(resource_known.items()):
                        try:
                            if not proc.is_running() or proc.create_time() != key[1]:
                                continue
                            times = proc.cpu_times()
                            total = times.user + times.system
                            current_cpu[key] = total
                            cpu_delta += max(0.0, total - previous_cpu.get(key, total))
                            rss += proc.memory_info().rss
                            count += 1
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            continue
                    row = {"type": "sample", "elapsed_seconds": now - started,
                           "cpu_percent": 100 * cpu_delta / (now - previous_sample) if previous_sample is not None else None,
                           "rss_bytes": rss, "observed_process_count": count}
                    previous_cpu, previous_sample = current_cpu, now
                    try:
                        gpu = subprocess.run(["nvidia-smi", "-i", ",".join(gpu_uuids),
                                              "--query-gpu=uuid,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                                             capture_output=True, text=True, timeout=5, check=True)
                        rows = list(csv.reader(gpu.stdout.strip().splitlines()))
                        values = {entry[0].strip(): {"memory_used_mib": float(entry[1]),
                                                    "utilization_percent": float(entry[2])} for entry in rows}
                        if set(values) != set(gpu_uuids) or len(rows) != 2:
                            raise RuntimeError("GPU identity changed during resource sampling")
                        row["gpus"] = values
                    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
                        row["gpu_sample_error"] = str(error)
                    emit(row)
                    next_sample = time.monotonic() + 2
                for key, _ in selector.select(timeout=0.2):
                    raw = os.read(key.fileobj.fileno(), 65536)
                    if not raw:
                        eof = True
                        selector.unregister(key.fileobj)
                    else:
                        text = raw.decode("utf-8", errors="replace")
                        logfile.write(raw)
                        logfile.flush()
                        # Preserve complete raw output on scratch. Forward only
                        # bounded timing lines, avoiding another unlimited copy
                        # in run_logged and the enclosing PBS command log.
                        console_buffer = (console_buffer + text)[-131072:]
                        while "\n" in console_buffer:
                            line, console_buffer = console_buffer.split("\n", 1)
                            if re.search(r"\bstep:\d+\b", line) and "timing_s/step:" in line:
                                print(line[:65536], flush=True)
                code = process.poll()
                if code is not None:
                    returncode = code if code >= 0 else 128 - code
                    exit_seen = exit_seen or time.monotonic()
                    if eof or time.monotonic() - exit_seen > 5:
                        break
        except Exception as error:
            emit({"type": "monitor_error", "error": f"{type(error).__name__}: {error}"})
            print(f"T00B_MONITOR_ERROR: {error}", file=sys.stderr, flush=True)
            returncode = 1
        finally:
            if process is not None:
                discover(process.pid, training_known)
                discover(args.server_pid, server_known, args.server_create_time)
                cleanup_known = {**training_known, **server_known}
                # Only this launch's group and identity-checked descendants.
                remaining = []
                group_verified = False
                for key, proc in cleanup_known.items():
                    try:
                        if proc.is_running() and proc.create_time() == key[1]:
                            if os.getpgid(proc.pid) == process.pid:
                                group_verified = True
                            proc.terminate()
                            remaining.append(proc)
                    except (ProcessLookupError, psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                if group_verified:
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                _, alive = psutil.wait_procs(remaining, timeout=8)
                for proc in alive:
                    try:
                        proc.kill()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                # A delivered KILL is asynchronous; allow verified descendants
                # to exit before declaring cleanup incomplete.
                psutil.wait_procs(alive, timeout=2)
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=3)
            cleanup_remaining = []
            for key, proc in {**training_known, **server_known}.items():
                try:
                    if proc.is_running() and proc.create_time() == key[1] and proc.status() != psutil.STATUS_ZOMBIE:
                        cleanup_remaining.append({"pid": key[0], "create_time": key[1]})
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            if cleanup_remaining and returncode == 0:
                returncode = 1
            # Capture the original completion clock before post-run evidence I/O.
            completed_elapsed = time.monotonic() - started
            worker_evidence_count = 0
            worker_evidence_error = None
            try:
                rows = worker_manifest_evidence(manifest)
                for row in rows:
                    emit(row)  # Exact JSON bytes (escaped) + SHA survive resource-log archival.
                worker_evidence_count = len(rows)
                if manifest.get("worker_compilation_cache") and not rows and returncode == 0:
                    raise ValueError("Successful training has no worker cache manifests")
            except Exception as error:
                worker_evidence_error = f"{type(error).__name__}: {error}"
                emit({"type": "worker_compilation_cache_error", "error": worker_evidence_error})
                if returncode == 0:
                    returncode = 1
            summary = {"type": "completion", "cleanup_remaining": cleanup_remaining, "elapsed_seconds": completed_elapsed,
                       "worker_cache_manifest_count": worker_evidence_count, "worker_cache_evidence_error": worker_evidence_error,
                       "returncode": returncode, "timed_out": timed_out,
                       "interrupted_signal": interrupted["signal"], "sampling_interval_seconds": 2,
                       "allocated_gpu_uuids": gpu_uuids}
            emit(summary)
            print("T00B_MONITOR_STATUS_JSON: " + json.dumps(summary), flush=True)
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())

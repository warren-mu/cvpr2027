"""Four-GPU packing plumbing smoke; never a VAGEN/research throughput result."""
import argparse
import datetime as dt
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
UUID = re.compile(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\Z")
NAME = re.compile(r"t00b_[A-Za-z0-9_]+\Z")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def write_result(name, metrics, config):
    save_new(ROOT / "results" / f"{name}.json", {"metrics": metrics, "config": config})
    print("FINAL_METRICS_JSON: " + json.dumps(metrics, sort_keys=True, allow_nan=False), flush=True)


def mask(value, count):
    result = value.split(",")
    if len(result) != count or len(set(result)) != count or any(not UUID.fullmatch(item) for item in result):
        raise ValueError(f"Expected exactly {count} unique full GPU UUIDs")
    return result


def prepare(args):
    original = mask(os.environ.get("CUDA_VISIBLE_DEVICES", ""), 4)
    pbs = os.environ["PBS_JOBID"]
    scratch = Path(os.environ["CVPR_SCRATCH"]).resolve()
    queue = args.queue_root.resolve()
    if scratch not in queue.parents:
        raise ValueError("Queue must be a private scratch subdirectory")
    for name in (f"{args.name}_gate", f"{args.name}_a", f"{args.name}_b"):
        for path in [ROOT / "results" / f"{name}.json", ROOT / "logs/runs" / name,
                     *[ROOT / "logs/runs" / f"{name}{suffix}" for suffix in (".log", ".meta.json", ".diff", ".worker.out")]]:
            if path.exists():
                raise FileExistsError(f"Prior evidence must be preserved: {path}")
    if (ROOT / "results" / f"{args.name}.json").exists():
        raise FileExistsError("Summary already exists; use a new name")
    queue.mkdir(parents=True, exist_ok=False)
    save_new(queue / "allocation.json", {"name": args.name, "pbs_jobid": pbs,
             "host": socket.gethostname(), "gpu_uuids": original,
             "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
             "requested_compute_seconds": args.seconds, "task_deadline_seconds": args.task_timeout,
             "worker_timeout_seconds": args.worker_timeout,
             "source_sha256": digest(__file__), "kind": "synthetic_packing_smoke_not_research"})
    return 0


def actual_cuda_identity(torch, rank, expected):
    properties = torch.cuda.get_device_properties(rank)
    value = getattr(properties, "uuid", None)
    value = str(value) if value is not None else None
    if value and not value.startswith("GPU-"):
        value = "GPU-" + value
    observations, error = [], None
    try:
        completed = subprocess.run(["nvidia-smi", "-i", expected, "--query-compute-apps=pid,gpu_uuid",
                                    "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10, check=True)
        for line in completed.stdout.splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) == 2 and fields[0] == str(os.getpid()):
                observations.append(fields[1])
    except (OSError, subprocess.SubprocessError) as failure:
        error = f"{type(failure).__name__}: {failure}"
    if value is not None and value != expected:
        raise RuntimeError(f"CUDA device properties UUID differs from assigned UUID: {value}")
    if observations and set(observations) != {expected}:
        raise RuntimeError("nvidia-smi process UUID differs from assigned UUID")
    if value != expected and set(observations) != {expected}:
        raise RuntimeError("Could not verify this rank's actual CUDA device UUID")
    return {"gpu_uuid": expected, "cuda_properties_uuid": value, "nvidia_smi_pid_gpu_uuids": observations,
            "nvidia_smi_error": error, "gpu_name": properties.name}


def rank_main(rank, config):
    path = Path(config["rank_directory"]) / f"rank_{rank}.json"
    log_path = path.with_suffix(".log")
    row = {"rank": rank, "pid": os.getpid(), "success": False, "completed_updates": 0,
           "assigned_gpu_uuid": config["gpu_uuids"][rank], "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    import contextlib
    with log_path.open("x", buffering=1) as output, contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
        torch = dist = None
        try:
            import torch
            import torch.distributed as dist
            torch.set_num_threads(1)
            if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
                raise RuntimeError("This task requires exactly two visible CUDA GPUs")
            torch.cuda.set_device(rank)
            torch.manual_seed(config["seed"])
            torch.cuda.manual_seed_all(config["seed"])
            device = torch.device("cuda", rank)
            # Two ranks really compute different data and average every gradient.
            model = torch.nn.Sequential(torch.nn.Linear(1024, 2048), torch.nn.ReLU(),
                                        torch.nn.Linear(2048, 1024)).to(device)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
            generator = torch.Generator(device=device).manual_seed(config["seed"] + rank + 1)
            inputs = torch.randn(256, 1024, device=device, generator=generator)
            target = torch.randn(256, 1024, device=device, generator=generator)
            initial = next(model.parameters()).detach().clone()
            row.update(actual_cuda_identity(torch, rank, config["gpu_uuids"][rank]))
            row.update(torch_version=torch.__version__, cuda_device_index=rank,
                       actual_tensor_device=str(inputs.device), parameter_count=sum(p.numel() for p in model.parameters()))
            store = dist.TCPStore("127.0.0.1", config["port"], None, False,
                                  timeout=dt.timedelta(seconds=45))
            dist.init_process_group("nccl", store=store, rank=rank, world_size=2,
                                    timeout=dt.timedelta(seconds=45))
            dist.barrier()
            torch.cuda.synchronize(device)
            started = time.monotonic_ns()
            row["compute_start_monotonic_ns"] = started
            row["compute_start_epoch_ns"] = time.time_ns()
            control = torch.ones(1, dtype=torch.int32, device=device)
            while True:
                if rank == 0:
                    control.fill_(int((time.monotonic_ns() - started) / 1e9 < config["seconds"] or row["completed_updates"] < 2))
                dist.broadcast(control, src=0)
                if not control.item():
                    break
                optimizer.zero_grad(set_to_none=True)
                prediction = model(inputs)
                loss = torch.nn.functional.mse_loss(prediction, target)
                loss.backward()
                if row["completed_updates"] == 0:
                    local = next(model.parameters()).grad.detach().double().sum()
                    local_checksums = [torch.zeros_like(local) for _ in range(2)]
                    dist.all_gather(local_checksums, local)
                for parameter in model.parameters():
                    dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
                    parameter.grad.div_(2)
                if row["completed_updates"] == 0:
                    averaged = next(model.parameters()).grad.detach().double().sum()
                    expected = torch.stack(local_checksums).mean()
                    error = float((averaged - expected).abs().item())
                    row["first_gradient_average"] = {"local_checksums": [float(item.item()) for item in local_checksums],
                        "observed_checksum": float(averaged.item()), "expected_checksum": float(expected.item()),
                        "absolute_error": error, "check_atol": 1e-5, "check_rtol": 1e-5,
                        "consistent": bool(torch.isclose(averaged, expected, atol=1e-5, rtol=1e-5).item())}
                    row["first_gradient_norm"] = float(next(model.parameters()).grad.norm().item())
                    row["first_loss"] = float(loss.item())
                optimizer.step()
                row["completed_updates"] += 1
                if row["completed_updates"] == 1:
                    print(json.dumps({"event": "first_real_cuda_update", "rank": rank,
                                      "gpu_uuid": row["gpu_uuid"], "loss": row["first_loss"]}), flush=True)
            torch.cuda.synchronize(device)
            row.update(compute_end_monotonic_ns=time.monotonic_ns(), compute_end_epoch_ns=time.time_ns(),
                       last_loss=float(loss.item()), parameter_delta_norm=float((next(model.parameters()).detach() - initial).norm().item()),
                       peak_cuda_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
                       allreduce_operations=row["completed_updates"] * len(list(model.parameters())))
            row["success"] = (row["completed_updates"] >= 2 and row["first_gradient_norm"] > 0
                              and row["parameter_delta_norm"] > 0 and row["first_gradient_average"]["consistent"]
                              and math.isfinite(row["first_loss"]) and math.isfinite(row["last_loss"]))
            if not row["success"]:
                raise RuntimeError("Actual gradient/update evidence did not pass")
            dist.barrier()
            print(json.dumps({"event": "completed", "rank": rank, "completed_updates": row["completed_updates"]}), flush=True)
        except BaseException as error:
            row.update(success=False, error=f"{type(error).__name__}: {error}")
            traceback.print_exc()
        finally:
            if dist is not None and dist.is_initialized():
                try:
                    dist.destroy_process_group()
                except BaseException as error:
                    row.update(success=False, cleanup_error=f"{type(error).__name__}: {error}")
            save_new(path, row)
    if not row["success"]:
        raise SystemExit(1)


def interval_overlap(intervals):
    if not intervals or any(len(row) != 2 or not all(type(item) is int for item in row) or row[1] <= row[0] for row in intervals):
        raise ValueError("Invalid measured compute intervals")
    return max(0, min(row[1] for row in intervals) - max(row[0] for row in intervals)) / 1e9


def train(args):
    config = {"name": args.name, "kind": "synthetic_packing_smoke_not_research", "seed": args.seed,
              "seconds": args.seconds, "timeout_seconds": args.timeout_seconds, "pbs_jobid": os.environ.get("PBS_JOBID"),
              "host": socket.gethostname(), "source_sha256": digest(__file__),
              "queue_job": os.environ.get("CVPR_QUEUE_JOB"), "queue_root": os.environ.get("CVPR_QUEUE_ROOT"),
              "parent_cuda_visible_devices": os.environ.get("CVPR_QUEUE_PARENT_MASK"),
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "backend": "nccl",
              "note": "Seeded synthetic SGD and gradient allreduce validate scheduling only; not VAGEN or scientific training performance."}
    metrics = {"passed": False, "ranks": [], "cleanup_errors": []}
    processes = []
    stopped = {"signal": None}
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    for sig in handlers:
        signal.signal(sig, lambda number, _frame: stopped.update(signal=number))
    try:
        config["gpu_uuids"] = mask(config["cuda_visible_devices"] or "", 2)
        parent = mask(config["parent_cuda_visible_devices"] or "", 4)
        if not set(config["gpu_uuids"]).issubset(parent) or config["queue_job"] != args.name or not config["pbs_jobid"]:
            raise ValueError("Child must originate from this four-GPU PBS queue")
        directory = ROOT / "logs/runs" / args.name / "ranks"
        directory.mkdir(parents=True, exist_ok=False)
        config["rank_directory"] = str(directory)
        import torch.distributed as dist
        store = dist.TCPStore("127.0.0.1", 0, None, True, timeout=dt.timedelta(seconds=45), wait_for_workers=False)
        config["port"] = store.port  # The parent holds this bound port until all ranks exit.
        deadline = time.monotonic() + args.timeout_seconds
        context = mp.get_context("spawn")
        for rank in range(2):
            process = context.Process(target=rank_main, args=(rank, config))
            process.start()
            processes.append(process)
        while any(process.is_alive() for process in processes):
            if stopped["signal"] is not None:
                raise InterruptedError(f"Received signal {stopped['signal']}")
            if time.monotonic() >= deadline:
                raise TimeoutError("Synthetic child deadline exceeded")
            if any(process.exitcode not in (None, 0) for process in processes):
                raise RuntimeError("A CUDA rank failed; stop its peer")
            time.sleep(0.1)
        for process in processes:
            process.join(timeout=0)
        if any(process.exitcode != 0 for process in processes):
            raise RuntimeError("CUDA rank exited nonzero")
    except BaseException as error:
        metrics["error"] = f"{type(error).__name__}: {error}"
        traceback.print_exc()
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
        deadline = time.monotonic() + 3
        for process in processes:
            process.join(timeout=max(0, deadline - time.monotonic()))
        for process in processes:
            if process.is_alive():
                process.kill()
        deadline = time.monotonic() + 2
        for process in processes:
            process.join(timeout=max(0, deadline - time.monotonic()))
            if process.is_alive():
                metrics["cleanup_errors"].append(f"Rank PID {process.pid} did not exit")
        metrics["rank_exit_codes"] = [process.exitcode for process in processes]
        directory = Path(config["rank_directory"]) if "rank_directory" in config else None
        for rank in range(2):
            path = directory / f"rank_{rank}.json" if directory else None
            if path and path.exists():
                row = read(path)
                row["evidence"] = {"file": str(path), "sha256": digest(path),
                                   "log": str(path.with_suffix('.log')), "log_sha256": digest(path.with_suffix('.log'))}
                metrics["ranks"].append(row)
        good = len(metrics["ranks"]) == 2 and all(row.get("success") is True for row in metrics["ranks"])
        if good:
            intervals = [(row["compute_start_monotonic_ns"], row["compute_end_monotonic_ns"]) for row in metrics["ranks"]]
            metrics["two_rank_compute_overlap_seconds"] = interval_overlap(intervals)
            metrics["compute_interval_monotonic_ns"] = [max(row[0] for row in intervals), min(row[1] for row in intervals)]
            metrics["actual_gpu_uuids"] = [row["gpu_uuid"] for row in metrics["ranks"]]
            metrics["passed"] = (not metrics.get("error") and not metrics["cleanup_errors"]
                                  and metrics["rank_exit_codes"] == [0, 0]
                                  and len(set(metrics["actual_gpu_uuids"])) == 2
                                  and metrics["two_rank_compute_overlap_seconds"] > 0)
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        write_result(args.name, metrics, config)
    return 0 if metrics["passed"] else 1


def load_logged(path):
    path = Path(path).resolve()
    data = read(path)
    meta, metrics = data.get("_meta", {}), data.get("metrics", {})
    if meta.get("producer") != "scripts/run_logged.py" or meta.get("metrics_sha256") != metrics_digest(metrics):
        raise ValueError(f"Unverified result provenance: {path}")
    if not re.fullmatch(r"[0-9a-f]{40}", str(meta.get("git_commit"))) or not meta.get("command"):
        raise ValueError(f"Missing committed producer command: {path}")
    log = ROOT / meta["log_path"]
    meta_path = ROOT / "logs/runs" / f"{meta['name']}.meta.json"
    if read(meta_path) != meta:
        raise ValueError(f"Result/log metadata differ: {path}")
    lines = log.read_text().splitlines()
    finals = [line.split("FINAL_METRICS_JSON: ", 1)[1] for line in lines if line.startswith("FINAL_METRICS_JSON: ")]
    if not finals or json.loads(finals[-1]) != metrics:
        raise ValueError(f"Missing or inconsistent FINAL_METRICS_JSON: {path}")
    if f"# exit_code: {meta['exit_code']}" not in lines:
        raise ValueError(f"Missing logged exit code: {path}")
    return data, {"file": str(path), "sha256": digest(path), "metrics_sha256": meta["metrics_sha256"],
                  "log": str(log), "log_sha256": digest(log), "meta": str(meta_path), "meta_sha256": digest(meta_path)}


def verify_pair(sources, allocation, worker):
    errors, intervals, devices, ports = [], [], [], []
    for index, data in enumerate(sources):
        config, metrics, meta = data["config"], data["metrics"], data["_meta"]
        expected_name = allocation["name"] + ("_a" if index == 0 else "_b")
        if config.get("name") != expected_name or meta.get("name") != expected_name or config.get("queue_job") != expected_name:
            errors.append("Child name/queue/provenance mismatch")
        if meta.get("exit_code") != 0 or metrics.get("passed") is not True:
            errors.append(f"Child {expected_name} did not pass with exit 0")
        if config.get("pbs_jobid") != allocation["pbs_jobid"] or config.get("host") != allocation["host"]:
            errors.append("Compute clocks do not originate from the same PBS host/job")
        if config.get("parent_cuda_visible_devices") != allocation["cuda_visible_devices"]:
            errors.append("Child original PBS mask mismatch")
        if (config.get("source_sha256") != allocation["source_sha256"]
                or config.get("kind") != "synthetic_packing_smoke_not_research"
                or config.get("backend") != "nccl"):
            errors.append("Child source identity or compute method differs from plan")
        command = meta.get("command", [])
        if command[1:4] != ["-m", "src.tools.pack_smoke", "train"]:
            errors.append("Child logged producer is not the committed synthetic CUDA task")
        ranks = metrics.get("ranks", [])
        actual = [row.get("gpu_uuid") for row in ranks]
        if len(ranks) != 2 or len(set(actual)) != 2 or not set(actual).issubset(allocation["gpu_uuids"]):
            errors.append("Each child must compute on exactly two distinct allocated GPU UUIDs")
        if config.get("gpu_uuids") != actual or metrics.get("actual_gpu_uuids") != actual:
            errors.append("Requested and actual rank GPU UUIDs differ")
        for rank, row in enumerate(ranks):
            if (row.get("rank") != rank or row.get("success") is not True or row.get("completed_updates", 0) < 2
                    or row.get("first_gradient_norm", 0) <= 0 or row.get("parameter_delta_norm", 0) <= 0
                    or row.get("first_gradient_average", {}).get("consistent") is not True
                    or row.get("allreduce_operations", 0) < row.get("completed_updates", 0)):
                errors.append("Missing measured real CUDA gradient/update evidence")
        devices.append(set(actual))
        try:
            raw_intervals = [(row["compute_start_monotonic_ns"], row["compute_end_monotonic_ns"]) for row in ranks]
            if interval_overlap(raw_intervals) <= 0:
                errors.append("Child ranks did not overlap")
            derived = [max(row[0] for row in raw_intervals), min(row[1] for row in raw_intervals)]
            if derived != metrics.get("compute_interval_monotonic_ns"):
                errors.append("Child compute interval differs from measured rank intervals")
            intervals.append(derived)
        except (KeyError, ValueError) as error:
            errors.append(f"Missing or invalid rank compute intervals: {error}")
            intervals.append([])
        ports.append(config.get("port"))
        record = next((row for row in worker.get("jobs", []) if row.get("name") == expected_name), {})
        if record.get("exit_code") != 0 or record.get("gpu_ids") != actual:
            errors.append("Queue completion/assignment differs from rank evidence")
    if len(sources) != 2:
        errors.append("Exactly two child results are required")
    overlap = None
    if len(sources) == 2:
        if devices[0] & devices[1] or devices[0] | devices[1] != set(allocation["gpu_uuids"]):
            errors.append("Child GPU UUID sets overlap or do not cover all four allocated GPUs")
        if not all(type(port) is int and port > 0 for port in ports) or len(set(ports)) != 2:
            errors.append("Child rendezvous ports are not distinct")
        try:
            overlap = interval_overlap(intervals)
            if overlap <= 0:
                errors.append("No measured period with both ranks of both tasks computing concurrently")
        except ValueError as error:
            errors.append(str(error))
    if worker.get("exit_code") != 0 or worker.get("cleanup_errors"):
        errors.append("Queue worker failed or cleanup is uncertain")
    return errors, overlap


def summarize(args):
    metrics = {"passed": False, "kind": "synthetic_packing_smoke_not_research", "errors": [],
               "gate_exit_code": args.gate_exit, "worker_exit_code": args.worker_exit,
               "children": [], "sources": []}
    config = {"name": args.name, "queue_root": str(args.queue_root), "source_sha256": digest(__file__),
              "note": "Checks allocation isolation, actual CUDA work, simultaneous compute intervals and independent provenance only; no VAGEN performance inference."}
    sources = []
    try:
        allocation = read(args.queue_root / "allocation.json")
        config["allocation"] = allocation
        gate, evidence = load_logged(ROOT / "results" / f"{args.name}_gate.json")
        metrics["sources"].append(evidence)
        if args.gate_exit != 0 or gate["_meta"].get("exit_code") != 0 or gate["metrics"].get("passed") is not True:
            metrics["errors"].append("Render gate did not pass; child GPU computation is not authorized")
        gate_devices = [row.get("observation", {}).get("gpu_identity", {}).get("gpu_uuid")
                        for row in gate["metrics"].get("devices", [])]
        if (len(gate_devices) != 4 or set(gate_devices) != set(allocation["gpu_uuids"])
                or gate["config"].get("pbs_jobid") != allocation["pbs_jobid"]
                or gate["config"].get("node") != allocation["host"]):
            metrics["errors"].append("Gate did not validate exactly this same four-GPU PBS allocation")
        for suffix in ("a", "b"):
            path = ROOT / "results" / f"{args.name}_{suffix}.json"
            try:
                data, evidence = load_logged(path)
                for rank in data["metrics"].get("ranks", []):
                    raw = rank["evidence"]
                    if digest(raw["file"]) != raw["sha256"] or digest(raw["log"]) != raw["log_sha256"]:
                        raise ValueError("Rank raw evidence hash mismatch")
                    if read(raw["file"]) != {key: value for key, value in rank.items() if key != "evidence"}:
                        raise ValueError("Rank raw evidence differs from result")
                sources.append(data)
                metrics["sources"].append(evidence)
                metrics["children"].append({"name": data["config"].get("name"), "exit_code": data["_meta"].get("exit_code"),
                                            "metrics": data["metrics"]})
            except Exception as error:
                metrics["errors"].append(f"{path.name}: {type(error).__name__}: {error}")
        worker = read(args.queue_root / "worker_summary.json") if (args.queue_root / "worker_summary.json").exists() else {}
        metrics["queue_worker"] = worker
        errors, overlap = verify_pair(sources, allocation, worker)
        metrics["errors"].extend(errors)
        metrics["all_four_gpu_compute_overlap_seconds"] = overlap
        if len(metrics["sources"]) == 3:
            logs = metrics["sources"][1:]
            if len({row["log"] for row in logs}) != 2 or len({row["log_sha256"] for row in logs}) != 2:
                metrics["errors"].append("Child log paths/hashes are not independent")
        for filename in ("allocation.json", "worker_claim.json", "worker_summary.json"):
            path = args.queue_root / filename
            if path.exists():
                metrics.setdefault("queue_evidence", []).append({"file": str(path), "sha256": digest(path)})
        # Preserve copies so queue evidence is reviewable after scratch retention.
        evidence_dir = ROOT / "logs/runs" / args.name / "queue_evidence"
        for path in sorted(args.queue_root.rglob("*.json")):
            relative = path.relative_to(args.queue_root)
            save_new(evidence_dir / relative, read(path))
        metrics["passed"] = not metrics["errors"] and args.gate_exit == args.worker_exit == 0
    except BaseException as error:
        metrics["errors"].append(f"{type(error).__name__}: {error}")
        traceback.print_exc()
    write_result(args.name, metrics, config)
    return 0 if metrics["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--queue-root", type=Path, required=True)
    prep.add_argument("--seconds", type=float, default=120)
    prep.add_argument("--task-timeout", type=float, default=240)
    prep.add_argument("--worker-timeout", type=float, default=360)
    child = sub.add_parser("train")
    child.add_argument("--seconds", type=float, default=120)
    child.add_argument("--timeout-seconds", type=float, default=240)
    child.add_argument("--seed", type=int, required=True)
    summary = sub.add_parser("summarize")
    summary.add_argument("--queue-root", type=Path, required=True)
    summary.add_argument("--gate-exit", type=int, required=True)
    summary.add_argument("--worker-exit", type=int, required=True)
    for command in (prep, child, summary):
        command.add_argument("--name", required=True)
    args = parser.parse_args()
    if not NAME.fullmatch(args.name):
        parser.error("Use a new t00b_ name containing letters, digits or underscores")
    if args.command in ("prepare", "train"):
        timeout = args.task_timeout if args.command == "prepare" else args.timeout_seconds
        if not (math.isfinite(args.seconds) and math.isfinite(timeout) and 1 <= args.seconds <= 180 and timeout >= args.seconds + 60):
            parser.error("Compute duration must be 1..180 seconds with at least 60 seconds startup/cleanup reserve")
        if args.command == "prepare" and not (math.isfinite(args.worker_timeout) and args.worker_timeout >= timeout + 30):
            parser.error("Worker timeout must leave 30 seconds beyond the child deadline")
    return {"prepare": prepare, "train": train, "summarize": summarize}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())

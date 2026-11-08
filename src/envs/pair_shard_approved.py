"""One explicit T01 scene on one PBS UUID; leaves the queued generator unchanged."""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import shutil
import traceback

from src.envs.thor_common import ROOT, interrupted, write_result
from src.tools.t01_partition_approved import DEFAULT_MANIFEST, audited_result, load_manifest, sha_file


def is_black_frame(measurement, resolution):
    return (measurement.get("shape") == [resolution, resolution, 3]
            and measurement.get("dtype") == "uint8" and measurement.get("nonzero_values") == 0)


def shard_worker(config, connection):
    """Keep per-scene review exports on scratch; compact journals stay portable."""
    from src.envs import pair_generator_approved as pair_generator
    from src.envs import pair_simulator
    config = dict(config)
    original_root, original_review = pair_generator.ROOT, pair_generator.review_image
    config["output_paths"] = {key: str(original_root / value) for key, value in config["output_paths"].items()}
    pair_generator.ROOT = Path(config["raw_evidence_dir"]) / "review_workspace"
    original_frame = pair_simulator.valid_frame
    original_source_frame = pair_generator.valid_frame

    def checked_frame(event, resolution, stage="capture"):
        frame = event.frame
        measurement = {"shape": list(frame.shape), "dtype": str(frame.dtype),
                       "min": int(frame.min()) if frame.size else None,
                       "max": int(frame.max()) if frame.size else None,
                       "nonzero_values": int((frame != 0).sum())}
        if is_black_frame(measurement, resolution):
            # SystemExit bypasses ordinary rejected-attempt handling. Its finally
            # still flushes the attempted row and saves the last observed frame.
            from PIL import Image
            directory = Path(config["raw_evidence_dir"])
            image = directory / "black_frame.png"
            Image.fromarray(frame).save(image)
            record = {"measurement": measurement, "image": str(image), "image_sha256": sha_file(image),
                      "stage": stage,
                      "scene": event.metadata.get("sceneName"), "last_action": event.metadata.get("lastAction"),
                      "reason": "measured uint8 RGB is entirely zero; infrastructure stop, no candidate acceptance"}
            with (directory / "black_frame.json").open("x") as stream:
                json.dump(record, stream, sort_keys=True)
            raise SystemExit(86)
        return original_frame(event, resolution)

    def review(path, row, pictures, target_ids):
        result = original_review(path, row, pictures, target_ids)
        row["review_image_sha256"] = sha_file(result)
        return result

    pair_generator.review_image = review
    pair_simulator.valid_frame = checked_frame
    pair_generator.valid_frame = lambda event, resolution: checked_frame(event, resolution, "source")
    try:
        return pair_generator.generator_worker(config, connection)
    finally:
        pair_generator.ROOT, pair_generator.review_image = original_root, original_review
        pair_simulator.valid_frame, pair_generator.valid_frame = original_frame, original_source_frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--partition", choices=("a", "b"), required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--cpu-list", required=True)
    parser.add_argument("--gate-result", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"pilot_pairs_[A-Za-z0-9_]+", args.name):
        parser.error("Use a unique pilot_pairs_ name")
    output = ROOT / "results" / f"{args.name}.json"
    if output.exists():
        raise FileExistsError("Prior shard result is preserved")
    metrics, code = {"kind": "t01_scene_collect", "status": "failed_before_generation", "acceptance_passed": False}, 1
    config = {"collect_only": False, "seed": 0, "pilot_scenes": [args.scene],
              "shard": {"partition": args.partition, "scene": args.scene, "pbs_jobid": os.environ.get("PBS_JOBID")}}
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, interrupted)
    try:
        manifest, identity = load_manifest(args.manifest)
        config["shard"]["manifest"] = identity
        if args.scene not in manifest["partitions"][args.partition]["scenes"]:
            raise ValueError("Scene is outside the declared partition")
        gate, gate_ref = audited_result(args.gate_result)
        if (gate["_meta"]["exit_code"] != 0 or gate["metrics"].get("passed") is not True
                or gate["config"].get("pbs_jobid") != os.environ.get("PBS_JOBID")
                or gate["config"].get("resolution") != 224 or gate["config"].get("segmentation") is not True):
            raise ValueError("A successful same-job RGB/segmentation gate is required")
        devices = gate["config"]["allocation"]["devices"]
        matches = [item for item in devices if item["gpu_uuid"] == args.gpu_uuid]
        parent_mask = os.environ.get("T01_PARENT_CUDA_VISIBLE_DEVICES", "")
        if (len(matches) != 1 or args.gpu_uuid not in parent_mask.split(",")
                or os.environ.get("CUDA_VISIBLE_DEVICES") != args.gpu_uuid):
            raise ValueError("Child CUDA mask must be one verified UUID from the original PBS mask")
        cpus = [int(value) for value in args.cpu_list.split(",")]
        if len(cpus) != 12 or len(set(cpus)) != 12 or not set(cpus) <= os.sched_getaffinity(0):
            raise ValueError("Each shard requires twelve distinct allocated CPUs")
        os.sched_setaffinity(0, cpus)
        shard = {"partition": args.partition, "scene": args.scene, "seed": manifest["seed"],
                 "manifest": identity, "gate_source": gate_ref, "pbs_jobid": os.environ["PBS_JOBID"],
                 "parent_cuda_visible_devices": parent_mask, "child_cuda_visible_devices": args.gpu_uuid,
                 "gpu_identity": matches[0], "cpu_affinity": sorted(os.sched_getaffinity(0)),
                 "instances_per_gpu": 1, "generator_source_sha256": sha_file(Path(__file__).with_name("pair_generator_approved.py")),
                 "configuration_method": "private process PILOT_SCENES tuple then unchanged generator.main; no shared file edits"}
        config["shard"] = shard
        print("T01_SHARD_CONFIG: " + json.dumps(shard, sort_keys=True), flush=True)
        from src.envs import pair_generator_approved as pair_generator
        original_scenes, original_worker = pair_generator.PILOT_SCENES, pair_generator.generator_worker
        try:
            pair_generator.PILOT_SCENES = (args.scene,)
            pair_generator.generator_worker = shard_worker
            code = pair_generator.main(["--name", args.name, "--seed", str(manifest["seed"]),
                "--max-attempts", str(manifest["maximum_attempts_per_scene"]),
                "--timeout-seconds", str(args.timeout_seconds)])
        finally:
            pair_generator.PILOT_SCENES = original_scenes
            pair_generator.generator_worker = original_worker
        result = json.loads(output.read_text())
        metrics, config = result["metrics"], result["config"]
        config["shard"] = shard
        config["shard"]["review_export_policy"] = "per-scene images on scratch; merger exports at most first ten unique passed/pending candidates per class"
        metrics.update(kind="t01_scene_collect", acceptance_passed=False)
        # Empty journals from source-frame/scene failure are evidence too.
        metrics["output_files"] = [{"path": path, "sha256": sha_file(ROOT / path)}
            for path in config["output_paths"].values() if (ROOT / path).exists()]
        failures = metrics.get("scene_failures", [])
        black = any((item.get("source_evidence") or {}).get("source_rgb", {}).get("nonzero_values") == 0
                    and (item.get("source_evidence") or {}).get("source_rgb", {}).get("shape") == [224, 224, 3]
                    for item in failures)
        metrics["source_black_frame_detected"] = black
        metrics["capture_black_frame_detected"] = False
        black_record = Path(config["raw_evidence_dir"]) / "black_frame.json"
        if black_record.exists():
            evidence = json.loads(black_record.read_text())
            if not is_black_frame(evidence["measurement"], 224) or sha_file(evidence["image"]) != evidence["image_sha256"]:
                raise ValueError("Black-frame stop evidence hash/schema mismatch")
            metrics["black_frame_stop"] = {"file": str(black_record), "sha256": sha_file(black_record), **evidence}
            portable = ROOT / "logs/runs" / args.name / "black_frame"
            portable.mkdir(parents=True, exist_ok=False)
            for key, source in (("portable_file", black_record), ("portable_image", Path(evidence["image"]))):
                destination = portable / source.name
                with source.open("rb") as incoming, destination.open("xb") as outgoing:
                    shutil.copyfileobj(incoming, outgoing)
                copied_hash = sha_file(destination)
                if copied_hash != sha_file(source):
                    raise ValueError("Portable black-frame evidence copy hash mismatch")
                metrics["black_frame_stop"].update({key: str(destination), key + "_sha256": copied_hash})
            metrics["source_black_frame_detected"] |= evidence.get("stage") == "source"
            metrics["capture_black_frame_detected"] = evidence.get("stage") == "capture"
            black = True
        metrics["black_frame_detected"] = black
        if black:
            code = 1 if metrics.get("cleanup_errors") else 86
    except BaseException as error:
        metrics.update(status="shard_failed", error=f"{type(error).__name__}: {error}", acceptance_passed=False)
        code = 1
        traceback.print_exc()
    finally:
        write_result(args.name, metrics, config)
    return code


if __name__ == "__main__":
    raise SystemExit(main())

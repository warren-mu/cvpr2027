"""Adapt an actually measured v2 black capture to portable retry evidence.

No simulator calls or result mutation. The shard parent calls this only after
its owned construction/replay processes have completed clean bounded cleanup.
"""
import hashlib
import json
from pathlib import Path
import re


def digest(body):
    return hashlib.sha256(body).hexdigest()


def checked_reference(reference, root, maximum):
    if not isinstance(reference, dict) or not re.fullmatch(r"[0-9a-f]{64}", str(reference.get("sha256", ""))):
        raise ValueError("Missing complete SHA-256 reference for a black-frame source")
    path = Path(reference["file"]).resolve()
    path.relative_to(root)
    if not path.is_file() or not 0 < path.stat().st_size <= maximum:
        raise ValueError("Black-frame evidence is missing or exceeds its small-file cap")
    body = path.read_bytes()
    if digest(body) != reference["sha256"] or (reference.get("bytes") is not None and reference["bytes"] != len(body)):
        raise ValueError("Black-frame source hash/size mismatch")
    return path, body


def portable_black_stop(reference, shard, repo):
    """Return the existing stop schema; never infer a black frame from missing data."""
    from PIL import Image
    import io

    repo, root = Path(repo).resolve(), Path(shard["raw_evidence_dir"]).resolve()
    name, scene = shard["name"], shard["scene"]
    identity, allocation = shard["gpu_identity"], shard["allocation"]
    if (not re.fullmatch(r"pilot_pairs_v2_[A-Za-z0-9_]+", name) or shard.get("resolution") != 255
            or shard.get("pbs_jobid") != allocation.get("pbs_jobid")
            or identity not in allocation.get("devices", []) or shard.get("logical_index") != identity.get("logical_index")
            or shard.get("child_cuda_visible_devices") != identity.get("gpu_uuid")
            or shard.get("parent_cuda_visible_devices") != allocation.get("cuda_visible_devices")
            or not shard.get("node")):
        raise ValueError("Black capture has no complete matching shard/allocation identity")
    failure_path, failure_body = checked_reference(reference, root, 1024 * 1024)
    failure = json.loads(failure_body)
    if failure.get("valid") is not True or failure.get("black") is not True or failure.get("rgb_shape") != [255, 255, 3]:
        raise ValueError("Backend did not record a correctly shaped all-black RGB capture")
    image_path, image_body = checked_reference(failure.get("files", {}).get("rgb"), root, 2 * 1024 * 1024)
    metadata_path, metadata_body = checked_reference(failure.get("files", {}).get("metadata"), root, 8 * 1024 * 1024)
    if image_path.parent != failure_path.parent or metadata_path.parent != failure_path.parent:
        raise ValueError("Failure, PNG and metadata are not from the same private backend capture directory")
    metadata = json.loads(metadata_body)
    # AI2-THOR 5 Server.create_event validates sequenceId on the top-level
    # packet, then constructs Event from agents[i]. Event.metadata therefore
    # need not contain sequenceId. Preserve that null; do not invent a value.
    # Controller.normalize_scene maps FloorPlanN to FloorPlanN_physics.
    # Accept only that documented spelling pair, retaining the exact native name.
    if (metadata.get("sceneName") not in (scene, scene + "_physics")
            or metadata.get("sequenceId") != failure.get("sequence_id")):
        raise ValueError("Black RGB metadata scene/optional sequence differs")
    with Image.open(io.BytesIO(image_body)) as image:
        if (image.mode != "RGB" or image.size != (255, 255)
                or image.getextrema() != ((0, 0), (0, 0), (0, 0))
                or digest(image.tobytes()) != failure.get("rgb_sha256")):
            raise ValueError("Decoded RGB is not the exact recorded all-zero uint8 frame")
    directory = repo / "logs/runs" / name / "black_frame"
    if directory.exists() or directory.is_symlink():
        raise FileExistsError("Preserve existing portable black evidence")
    directory.mkdir(parents=True, exist_ok=False)

    def copy_bytes(label, original, body):
        destination = directory / label
        with destination.open("xb") as stream:
            stream.write(body)
        if digest(destination.read_bytes()) != digest(body):
            raise ValueError("Portable black evidence copy differs from original")
        return {"original_file": str(original), "file": str(destination.relative_to(repo)), "sha256": digest(body), "bytes": len(body)}

    original_refs = {"failure": copy_bytes("backend_failure.json", failure_path, failure_body),
                     "metadata": copy_bytes("metadata.json", metadata_path, metadata_body),
                     "rgb": copy_bytes("black.png", image_path, image_body)}
    stage = "source" if failure.get("source_action") == "constructor" else "capture"
    record = {"measurement": {"shape": [255, 255, 3], "dtype": "uint8", "nonzero_values": 0, "min": 0, "max": 0},
              "image": str(image_path), "image_sha256": digest(image_body), "stage": stage,
              "scene": scene, "metadata_scene_name": metadata.get("sceneName"), "sequence_id": failure.get("sequence_id"),
              "last_action": metadata.get("lastAction"), "pbs_jobid": shard["pbs_jobid"],
              "node": shard["node"], "gpu_uuid": identity["gpu_uuid"], "backend_sources": original_refs,
              "reason": "decoded full uint8 RGB is all zero; infrastructure failure, never pair acceptance"}
    raw = root / "verified_black_frame.json"
    body = (json.dumps(record, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with raw.open("xb") as stream:
        stream.write(body)
    portable = copy_bytes("black_frame.json", raw, body)
    return {**record, "file": str(raw), "sha256": digest(body),
            "portable_file": portable["file"], "portable_file_sha256": portable["sha256"],
            "portable_image": original_refs["rgb"]["file"], "portable_image_sha256": original_refs["rgb"]["sha256"]}

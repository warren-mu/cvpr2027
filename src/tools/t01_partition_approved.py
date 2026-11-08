"""Versioned approved pilot manifest; immutable scene ownership matches v1."""
import json
from pathlib import Path
from src.tools.t01_partition import (ROOT, audited_result, canonical_hash, candidate_identity,
                                    candidate_outcome, cpu_shards, ensure_unused, scene_shards, sha_file)
DEFAULT_MANIFEST = ROOT / "configs/t01_parallel_manifest_approved_v2.json"
C_MANIFEST = ROOT / "configs/t01_parallel_manifest_approved_v3.json"


def require_launch_partition(manifest, partition):
    """v2 remains A/B; explicit v3 authorizes C under the same legacy owner."""
    permitted = {2: ("a", "b"), 3: ("legacy",)}.get(manifest.get("schema_version"), ())
    if partition not in permitted:
        raise ValueError("A/B require approved v2; the two-GPU legacy-six C requires explicit approved v3")
    return manifest["partitions"][partition]


def require_same_scene_ownership(first, second):
    """GPU topology may differ; ordered scene ownership and holdout may not."""
    ownership = lambda value: {key: group["scenes"] for key, group in value["partitions"].items()}
    if ownership(first) != ownership(second) or first["test_scenes"] != second["test_scenes"]:
        raise ValueError("Manifest versions change scene ownership/order or the fixed holdout")

def load_manifest(path):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    expected = {f"FloorPlan{i}" for i in (*range(1, 31), *range(201, 231))}
    groups = manifest["partitions"]
    flattened = [scene for group in groups.values() for scene in group["scenes"]]
    version = manifest.get("schema_version")
    if (type(version) is not int or version not in (2, 3)
            or manifest.get("manifest_id") != f"t01_parallel_approved_v{version}" or manifest.get("seed") != 0
            or manifest.get("collect_only") is not False or manifest.get("acceptance_policy") != "t01_human_20261108" or set(groups) != {"legacy", "a", "b"}
            or len(flattened) != len(set(flattened)) or set(flattened) != expected):
        raise ValueError("Manifest must partition all 60 fixed scenes without overlap, seed zero, approved human policy")
    legacy = ["FloorPlan1", "FloorPlan2", "FloorPlan201", "FloorPlan202", "FloorPlan25", "FloorPlan225"]
    if groups["legacy"]["scenes"] != legacy or groups["legacy"]["gpus"] != (2 if version == 3 else 1):
        raise ValueError("Legacy reserved scenes or versioned GPU count changed")
    if version == 3 and manifest.get("launch_partition") != "legacy":
        raise ValueError("Approved v3 is explicitly limited to the legacy-six owner")
    for key, numbers, gpus in (("a", range(1, 31), 2), ("b", range(201, 231), 4)):
        scenes = [f"FloorPlan{i}" for i in numbers if f"FloorPlan{i}" not in legacy]
        if groups[key]["scenes"] != scenes or groups[key]["gpus"] != gpus:
            raise ValueError(f"Unexpected fixed scene order or GPU count in partition {key}")
    if (set(manifest["test_scenes"]) != {f"FloorPlan{i}" for i in (*range(25, 31), *range(225, 231))}
            or manifest.get("cpu_cores_per_gpu") != 12 or manifest.get("simultaneous_thor_instances_per_gpu") != 1
            or manifest.get("maximum_attempts_per_scene") != 40):
        raise ValueError("Manifest split, affinity or collection policy changed")
    return manifest, {"path": str(path), "sha256": sha_file(path), "manifest_id": manifest["manifest_id"],
                      "schema_version": version}


"""Pure manifest, provenance and identity helpers for explicit T01 shards."""
import hashlib
import importlib.util
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "configs/t01_parallel_manifest_v1.json"


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def load_manifest(path):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    expected = {f"FloorPlan{i}" for i in (*range(1, 31), *range(201, 231))}
    groups = manifest["partitions"]
    flattened = [scene for group in groups.values() for scene in group["scenes"]]
    if (manifest.get("schema_version") != 1 or manifest.get("seed") != 0
            or manifest.get("collect_only") is not True or set(groups) != {"legacy", "a", "b"}
            or len(flattened) != len(set(flattened)) or set(flattened) != expected):
        raise ValueError("Manifest must partition all 60 fixed scenes without overlap, seed zero, collect-only")
    legacy = ["FloorPlan1", "FloorPlan2", "FloorPlan201", "FloorPlan202", "FloorPlan25", "FloorPlan225"]
    if groups["legacy"]["scenes"] != legacy:
        raise ValueError("Legacy reserved scenes changed")
    for key, numbers, gpus in (("a", range(1, 31), 2), ("b", range(201, 231), 4)):
        scenes = [f"FloorPlan{i}" for i in numbers if f"FloorPlan{i}" not in legacy]
        if groups[key]["scenes"] != scenes or groups[key]["gpus"] != gpus:
            raise ValueError(f"Unexpected fixed scene order or GPU count in partition {key}")
    if (set(manifest["test_scenes"]) != {f"FloorPlan{i}" for i in (*range(25, 31), *range(225, 231))}
            or manifest.get("cpu_cores_per_gpu") != 12 or manifest.get("simultaneous_thor_instances_per_gpu") != 1
            or manifest.get("maximum_attempts_per_scene") != 40):
        raise ValueError("Manifest split, affinity or collection policy changed")
    return manifest, {"path": str(path), "sha256": sha_file(path), "manifest_id": manifest["manifest_id"]}


def scene_shards(manifest, partition):
    group = manifest["partitions"][partition]
    return [group["scenes"][index::group["gpus"]] for index in range(group["gpus"])]


def cpu_shards(allowed, count):
    cpus = sorted(set(allowed))
    if len(cpus) < 12 * count:
        raise ValueError(f"Need {12 * count} allocated affinity CPUs, found {len(cpus)}")
    return [cpus[index * 12:(index + 1) * 12] for index in range(count)]


def ensure_unused(name):
    if not re.fullmatch(r"pilot_pairs_[A-Za-z0-9_]+", name):
        raise ValueError("Use a unique pilot_pairs_ name with only letters, digits and underscores")
    paths = [ROOT / "results" / f"{name}.json", ROOT / "logs/runs" / name]
    paths += [ROOT / "logs/runs" / f"{name}{suffix}" for suffix in (".log", ".meta.json", ".diff")]
    if any(path.exists() for path in paths):
        raise FileExistsError(f"Prior evidence exists for {name}; choose a new name")


def audited_result(path):
    path = (ROOT / path).resolve()
    relative = path.relative_to(ROOT)
    if relative.parts[0] != "results" or path.suffix != ".json":
        raise ValueError("Expected an explicit repository results JSON")
    raw = path.read_bytes()
    data = json.loads(raw)
    spec = importlib.util.spec_from_file_location("t01_unchanged_auditor", ROOT / "scripts/audit_provenance.py")
    auditor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(auditor)
    audit = auditor.audit_file(str(path))
    if audit["fail"]:
        raise ValueError(f"{relative}: {audit['fail']}")
    meta = data["_meta"]
    if type(meta.get("exit_code")) is not int or canonical_hash(data["metrics"]) != meta.get("metrics_sha256"):
        raise ValueError(f"Incomplete/hash-invalid source: {relative}")
    log = ROOT / (meta.get("log_path") or f"logs/runs/{meta['name']}.log")
    final = re.findall(r"^FINAL_METRICS_JSON:\s*(.+?)\s*$", log.read_text(errors="replace"), re.M)
    if not final or canonical_hash(json.loads(final[-1])) != meta["metrics_sha256"]:
        raise ValueError(f"Missing or mismatched FINAL: {relative}")
    return data, {"file": relative.as_posix(), "sha256": hashlib.sha256(raw).hexdigest(),
                  "metrics_sha256": meta["metrics_sha256"], "exit_code": meta["exit_code"],
                  "git_commit": meta.get("git_commit"), "audit": audit}


def candidate_identity(row):
    """Same physical pair proposal across retries counts once; source failures stay."""
    required = ("scene", "seed", "task", "w0", "w1")
    if any(key not in row for key in required) or any(
            any(field not in row[key] for field in ("poses", "openness")) for key in ("w0", "w1")):
        return None
    return canonical_hash({"scene": row["scene"], "seed": row["seed"], "task": row["task"],
        "w0_poses": sorted(row["w0"]["poses"], key=lambda pose: pose["objectName"]),
        "w1_poses": sorted(row["w1"]["poses"], key=lambda pose: pose["objectName"]),
        "w0_openness": row["w0"]["openness"], "w1_openness": row["w1"]["openness"],
        "viewpoints": row.get("viewpoints"), "start": row.get("start"),
        "common_observation_action": row.get("common_observation_action")})


def candidate_outcome(row):
    """Preserve disagreement among retry measurements; never choose a best one."""
    fields = ("kind", "status", "discard_reasons", "validity", "same_input_at_t", "same_input_at_t_strict",
              "prefix_comparison", "prefix_max_abs_diff", "evidence_comparison", "evidence_pixels_after_look",
              "target_name_w0", "target_name_w1", "restore_deterministic", "valid_physical", "targets_separable",
              "witness_actions_w0", "witness_actions_w1", "independent_check_failures")
    return canonical_hash({key: row.get(key) for key in fields})

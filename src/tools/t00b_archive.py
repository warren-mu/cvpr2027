"""Preserve scratch evidence as bounded, lossless gzip shards in the repo.

Raw files and intermediate compressed files are retained on scratch. A shard
contains consecutive raw bytes, so concatenating decompressed shards reproduces
the source exactly (including line breaks). No simulator or numerical analysis.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil

RAW_CHUNK_BYTES = 16 * 1024 * 1024
MAX_GZIP_BYTES = 32 * 1024 * 1024
NAMES = ("train.log", "server.log", "resources.jsonl", "frame_events.jsonl")


def sha256(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(data)
    return value.hexdigest()


def archive_file(source, destination, scratch_directory, chunk_bytes=RAW_CHUNK_BYTES):
    if not 0 < chunk_bytes <= RAW_CHUNK_BYTES:
        raise ValueError("Invalid evidence shard size")
    destination.mkdir(parents=True, exist_ok=True)
    scratch_directory.mkdir(parents=True, exist_ok=True)
    full_hash, size, shards = hashlib.sha256(), 0, []
    with source.open("rb") as stream:
        while True:
            data = stream.read(chunk_bytes)
            if not data and shards:
                break
            full_hash.update(data)
            name = f"{source.name}.part{len(shards):05d}.gz"
            compressed = scratch_directory / name
            with compressed.open("xb") as output:
                with gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=0) as zipped:
                    zipped.write(data)
            if compressed.stat().st_size > MAX_GZIP_BYTES:
                raise RuntimeError("Compressed evidence shard exceeds the 32 MiB repository limit; retained on scratch")
            target = destination / name
            with compressed.open("rb") as original, target.open("xb") as copy:
                shutil.copyfileobj(original, copy, 1024 * 1024)
            shards.append({"index": len(shards), "path": str(target), "raw_offset_bytes": size,
                           "raw_bytes": len(data), "raw_sha256": hashlib.sha256(data).hexdigest(),
                           "gzip_bytes": target.stat().st_size, "gzip_sha256": sha256(target)})
            size += len(data)
            if not data:
                break
    return {"source_path": str(source), "source_bytes": size, "source_sha256": full_hash.hexdigest(),
            "shard_count": len(shards), "shards": shards, "complete": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    args = parser.parse_args()
    result = {"format": "ordered-gzip-byte-shards-v1", "raw_chunk_bytes": RAW_CHUNK_BYTES,
              "max_repository_shard_bytes": MAX_GZIP_BYTES, "files": {}, "errors": []}
    names = NAMES + (("rgb_failure.json",) if (args.raw_dir / "rgb_failure.json").exists() else ())
    for name in names:
        source = args.raw_dir / name
        if not source.exists():
            result["files"][name] = {"source_path": str(source), "complete": False, "missing": True}
            continue
        try:
            result["files"][name] = archive_file(source, args.archive_dir, args.raw_dir / "evidence_gzip")
        except Exception as error:
            result["files"][name] = {"source_path": str(source), "complete": False}
            result["errors"].append(f"{name}: {type(error).__name__}: {error}")
    result["complete"] = not result["errors"] and all(item["complete"] for item in result["files"].values())
    args.index.parent.mkdir(parents=True, exist_ok=True)
    with args.index.open("x") as output:
        json.dump(result, output, indent=2)
        output.write("\n")
    print("T00B_EVIDENCE_ARCHIVE: " + json.dumps({"index": str(args.index), "complete": result["complete"],
                                               "errors": result["errors"]}), flush=True)
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Synthetic PNG/JSON evidence fixtures, never an actual rendering experiment."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from PIL import Image

from src.tools.t01v2_black_evidence import portable_black_stop


def reference(path):
    return {"file": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}


def fixture(root, pixels=(0, 0, 0)):
    raw = root / "scratch/shard/construction"
    raw.mkdir(parents=True)
    image, metadata, failure = raw / "rgb.png", raw / "metadata.json", raw / "failure.json"
    frame = Image.new("RGB", (255, 255), pixels)
    frame.save(image)
    metadata.write_text(json.dumps({"sceneName": "FloorPlan1", "sequenceId": 7, "lastAction": "Pass"}))
    data = {"valid": True, "black": True, "rgb_shape": [255, 255, 3], "source_action": "Pass", "sequence_id": 7,
            "rgb_sha256": hashlib.sha256(frame.tobytes()).hexdigest(),
            "files": {"rgb": reference(image), "metadata": reference(metadata)}}
    failure.write_text(json.dumps(data))
    device = {"logical_index": 0, "gpu_uuid": "GPU-11111111-1111-1111-1111-111111111111"}
    allocation = {"pbs_jobid": "123.hopper-m-02", "devices": [device], "cuda_visible_devices": device["gpu_uuid"]}
    shard = {"name": "pilot_pairs_v2_fixture", "scene": "FloorPlan1", "resolution": 255,
             "pbs_jobid": allocation["pbs_jobid"], "node": "hopper-11", "gpu_identity": device,
             "logical_index": 0, "allocation": allocation, "parent_cuda_visible_devices": device["gpu_uuid"],
             "child_cuda_visible_devices": device["gpu_uuid"], "raw_evidence_dir": str(raw.parent)}
    return reference(failure), shard, data


class V2PortableBlackTests(unittest.TestCase):
    def test_exact_black_png_produces_portable_watch_record_without_rewriting_sources(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            source, shard, _ = fixture(root)
            original = Path(source["file"]).read_bytes()
            stop = portable_black_stop(source, shard, root)
            self.assertEqual(stop["measurement"]["nonzero_values"], 0)
            self.assertEqual(stop["stage"], "capture")
            self.assertEqual(Path(source["file"]).read_bytes(), original)
            portable = root / stop["portable_file"]
            self.assertEqual(hashlib.sha256(portable.read_bytes()).hexdigest(), stop["sha256"])
            self.assertEqual(hashlib.sha256((root / stop["portable_image"]).read_bytes()).hexdigest(), stop["image_sha256"])
            with self.assertRaises(FileExistsError):
                portable_black_stop(source, shard, root)

    def test_rehashed_nonblack_or_incomplete_capture_is_never_black_proof(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            source, shard, _ = fixture(root, (1, 0, 0))
            with self.assertRaises(ValueError):
                portable_black_stop(source, shard, root)
        for field, value in (("valid", False), ("black", False), ("rgb_shape", [224, 224, 3]),
                             ("sequence_id", None), ("rgb_sha256", "0" * 64)):
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp).resolve()
                source, shard, data = fixture(root)
                data[field] = value
                Path(source["file"]).write_text(json.dumps(data))
                with self.subTest(field=field), self.assertRaises(ValueError):
                    portable_black_stop(reference(Path(source["file"])), shard, root)

    def test_real_event_metadata_may_omit_packet_level_sequence_id(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            source, shard, data = fixture(root)
            metadata = Path(data["files"]["metadata"]["file"])
            value = json.loads(metadata.read_text())
            del value["sequenceId"]
            value["sceneName"] = "FloorPlan1_physics"
            metadata.write_text(json.dumps(value))
            data["files"]["metadata"] = reference(metadata)
            data["sequence_id"] = None
            Path(source["file"]).write_text(json.dumps(data))
            stop = portable_black_stop(reference(Path(source["file"])), shard, root)
            self.assertIsNone(stop["sequence_id"])
            self.assertEqual(stop["metadata_scene_name"], "FloorPlan1_physics")
            self.assertEqual(stop["measurement"]["nonzero_values"], 0)

    def test_wrong_uuid_scene_source_hash_and_path_escape_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            source, shard, data = fixture(root)
            for key, value in (("child_cuda_visible_devices", "GPU-unallocated"), ("scene", "FloorPlan2"),
                               ("pbs_jobid", "999.hopper-m-02")):
                with self.subTest(key=key), self.assertRaises(ValueError):
                    portable_black_stop(source, {**shard, key: value}, root)
            with self.assertRaises(ValueError):
                portable_black_stop({**source, "sha256": "0" * 64}, shard, root)
            other = root / "outside.png"
            other.write_bytes(Path(data["files"]["rgb"]["file"]).read_bytes())
            data["files"]["rgb"] = reference(other)
            Path(source["file"]).write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                portable_black_stop(reference(Path(source["file"])), shard, root)


if __name__ == "__main__":
    unittest.main()

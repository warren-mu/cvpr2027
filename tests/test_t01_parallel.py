"""Synthetic CPU fixtures for partition safety and retry identity; no GPU imports."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from src.envs.pair_shard import is_black_frame
from src.tools.merge_pair_shards import JournalChunks, retry_roots
from src.tools.t01_partition import (ROOT, DEFAULT_MANIFEST, candidate_identity, candidate_outcome,
                                    cpu_shards, load_manifest, scene_shards, sha_file)


def pair_fixture():
    pose = {"objectName": "Cup_1", "position": {"x": 0.0, "y": 1.0, "z": 0.0},
            "rotation": {"x": 0.0, "y": 0.0, "z": 0.0}}
    return {"scene": "FloorPlan3", "seed": 0, "task": {"target_type": "Cup", "relation": "in", "anchor_type": "Cabinet"},
            "w0": {"poses": [copy.deepcopy(pose)], "openness": {"Cabinet_1": 0.0}},
            "w1": {"poses": [copy.deepcopy(pose)], "openness": {"Cabinet_1": 0.0}},
            "start": {"viewpoint": "v0", "yaw": 0}, "common_observation_action": "turn_left",
            "kind": "look_stay", "status": "collected", "same_input_at_t_strict": False,
            "prefix_max_abs_diff": 2, "validity": {"valid_physical": None, "witness_replay": False}}


class ParallelSafetyTests(unittest.TestCase):
    def test_fixed_sixty_scene_partition_and_holdout(self):
        manifest, _ = load_manifest(DEFAULT_MANIFEST)
        groups = [group["scenes"] for group in manifest["partitions"].values()]
        all_scenes = [scene for group in groups for scene in group]
        self.assertEqual(len(all_scenes), 60)
        self.assertEqual(len(set(all_scenes)), 60)
        self.assertEqual(len(manifest["test_scenes"]), 12)
        for partition, count in (("a", 2), ("b", 4)):
            shards = scene_shards(manifest, partition)
            self.assertEqual(len(shards), count)
            self.assertTrue(all(shards))
            self.assertEqual({scene for shard in shards for scene in shard}, set(manifest["partitions"][partition]["scenes"]))
            self.assertFalse(set(manifest["partitions"]["legacy"]["scenes"]) & {scene for shard in shards for scene in shard})

    def test_reassigning_reserved_scene_is_rejected(self):
        manifest = json.loads(DEFAULT_MANIFEST.read_text())
        manifest["partitions"]["a"]["scenes"][0] = "FloorPlan1"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                load_manifest(path)

    def test_affinity_uses_only_disjoint_allocated_twelve_cpu_sets(self):
        allowed = list(range(8, 80, 2)) + list(range(101, 113))
        shards = cpu_shards(allowed, 4)
        self.assertTrue(all(len(group) == 12 for group in shards))
        flattened = [cpu for group in shards for cpu in group]
        self.assertEqual(len(set(flattened)), 48)
        self.assertTrue(set(flattened) <= set(allowed))
        with self.assertRaises(ValueError):
            cpu_shards(allowed[:-1], 4)

    def test_retry_identity_ignores_transient_ids_but_not_physical_edit(self):
        row = pair_fixture()
        retry = copy.deepcopy(row)
        retry.update(attempt_id="different", target_w0="Cup|new-position-id", generation_seconds=999)
        retry["w0"]["key"] = "different_attempt_w0"
        self.assertEqual(candidate_identity(row), candidate_identity(retry))
        retry["w1"]["poses"][0]["position"]["x"] = 0.5
        self.assertNotEqual(candidate_identity(row), candidate_identity(retry))

    def test_container_openness_is_part_of_world_identity_and_unknown_is_not_empty(self):
        row = pair_fixture()
        changed = copy.deepcopy(row)
        changed["w1"]["openness"]["Cabinet_1"] = 1.0
        self.assertNotEqual(candidate_identity(row), candidate_identity(changed))
        del changed["w1"]["openness"]
        self.assertIsNone(candidate_identity(changed))

    def test_equal_status_does_not_hide_conflicting_measurements(self):
        row = pair_fixture()
        retry = copy.deepcopy(row)
        retry["generation_seconds"] = 12
        self.assertEqual(candidate_outcome(row), candidate_outcome(retry))
        retry["validity"]["witness_replay"] = True
        self.assertEqual(candidate_identity(row), candidate_identity(retry))
        self.assertNotEqual(candidate_outcome(row), candidate_outcome(retry))

    def test_retry_links_must_be_explicit_and_acyclic(self):
        a, b, c = [str(ROOT / "results" / f"fixture_{name}.json") for name in "abc"]
        roots = retry_roots({a, b, c}, [f"{b}={a}", f"{c}={b}"])
        self.assertEqual(set(roots.values()), {a})
        self.assertEqual(retry_roots({a, b}, [])[a], a)
        with self.assertRaises(ValueError):
            retry_roots({a, b}, [f"{a}={b}", f"{b}={a}"])
        with self.assertRaises(ValueError):
            retry_roots({a, b}, [f"{b}={c}"])

    def test_black_stop_requires_measured_shape_dtype_and_all_zero(self):
        measured = {"shape": [224, 224, 3], "dtype": "uint8", "nonzero_values": 0}
        self.assertTrue(is_black_frame(measured, 224))
        for field, value in (("shape", [300, 300, 3]), ("dtype", "float32"), ("nonzero_values", 1)):
            self.assertFalse(is_black_frame({**measured, field: value}, 224))
        self.assertFalse(is_black_frame({}, 224))

    def test_chunk_ledger_retains_failures_and_refuses_evidence_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            writer = JournalChunks(root, limit=80)
            rows = [{"status": "rejected", "reason": "physics_unknown"}, {"status": "collected", "accepted": False}]
            for row in rows:
                writer.write(row)
            records = writer.close()
            recovered = [json.loads(line) for ref in records for line in Path(ref["path"]).read_text().splitlines()]
            self.assertEqual(recovered, rows)
            self.assertTrue(all(ref["bytes"] <= 80 and sha_file(ref["path"]) == ref["sha256"] for ref in records))
            with self.assertRaises(FileExistsError):
                JournalChunks(root, limit=80).write(rows[0])


if __name__ == "__main__":
    unittest.main()

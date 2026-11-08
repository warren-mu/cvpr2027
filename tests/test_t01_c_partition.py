"""Synthetic CPU checks for the explicit C takeover; no PBS/GPU or real evidence."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from src.envs.pair_approved_policy import POLICY
from src.tools import merge_pair_approved as merger
from src.tools.t01_partition import DEFAULT_MANIFEST as V1_MANIFEST, load_manifest as load_v1
from src.tools.t01_partition_approved import (C_MANIFEST, DEFAULT_MANIFEST, load_manifest,
    require_launch_partition, require_same_scene_ownership, scene_shards, cpu_shards)


class CPartitionTests(unittest.TestCase):
    def versions(self):
        v2, ref2 = load_manifest(DEFAULT_MANIFEST)
        v3, ref3 = load_manifest(C_MANIFEST)
        v1, ref1 = load_v1(V1_MANIFEST)
        return v1, v2, v3, ref1, ref2, ref3

    def test_C_uses_exact_legacy_six_on_two_cards_and_preserves_default_A_B(self):
        v1, v2, v3, *_ = self.versions()
        self.assertEqual(DEFAULT_MANIFEST.name, "t01_parallel_manifest_approved_v2.json")
        require_same_scene_ownership(v1, v2)
        require_same_scene_ownership(v2, v3)
        self.assertEqual(v2["partitions"]["legacy"]["gpus"], 1)
        self.assertEqual(v3["partitions"]["legacy"]["gpus"], 2)
        self.assertEqual(v2["partitions"]["a"], v3["partitions"]["a"])
        self.assertEqual(v2["partitions"]["b"], v3["partitions"]["b"])
        self.assertEqual(scene_shards(v3, "legacy"), [
            ["FloorPlan1", "FloorPlan201", "FloorPlan25"],
            ["FloorPlan2", "FloorPlan202", "FloorPlan225"]])
        cpus = cpu_shards(range(24, 48), 2)
        self.assertEqual([len(group) for group in cpus], [12, 12])
        self.assertFalse(set(cpus[0]) & set(cpus[1]))
        self.assertEqual(v3["simultaneous_thor_instances_per_gpu"], 1)

    def test_C_requires_explicit_v3_and_A_B_remain_v2(self):
        _, v2, v3, *_ = self.versions()
        for partition in ("a", "b"):
            require_launch_partition(v2, partition)
            with self.assertRaises(ValueError):
                require_launch_partition(v3, partition)
        require_launch_partition(v3, "legacy")
        with self.assertRaises(ValueError):
            require_launch_partition(v2, "legacy")

    def test_manifest_scene_order_holdout_or_C_topology_changes_are_rejected(self):
        _, v2, v3, *_ = self.versions()
        mutations = (
            lambda value: value["partitions"]["legacy"]["scenes"].reverse(),
            lambda value: value["partitions"]["a"]["scenes"].__setitem__(0, "FloorPlan1"),
            lambda value: value["test_scenes"].__setitem__(0, "FloorPlan1"),
            lambda value: value["partitions"]["legacy"].update(gpus=1),
            lambda value: value.update(manifest_id="t01_parallel_approved_v2"),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            for mutate in mutations:
                changed = copy.deepcopy(v3)
                mutate(changed)
                path.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):
                    load_manifest(path)
            moved = copy.deepcopy(v3)
            moved["partitions"]["legacy"]["scenes"][0] = "FloorPlan3"
            with self.assertRaises(ValueError):
                require_same_scene_ownership(v2, moved)

    def test_mixed_exact_manifest_hashes_are_accepted_only_for_their_owner(self):
        _, _, _, ref1, ref2, ref3 = self.versions()
        catalog = [ref2, ref3]
        for ref, partition in ((ref2, "a"), (ref2, "b"), (ref3, "legacy")):
            # Existing v2 records have no schema_version field; the trusted
            # catalog, not source-supplied version labels, determines permission.
            source_ref = {key: ref[key] for key in ("sha256", "manifest_id")}
            self.assertEqual(merger.require_source_manifest(source_ref, False, catalog, ref1,
                             partition=partition)["sha256"], ref["sha256"])
        bad_cases = (({**ref3, "sha256": "0" * 64}, "legacy"),
                     ({**ref3, "manifest_id": ref2["manifest_id"]}, "legacy"),
                     (ref2, "legacy"), (ref3, "a"), (ref1, "legacy"))
        for ref, partition in bad_cases:
            with self.assertRaises(ValueError):
                merger.require_source_manifest(ref, False, catalog, ref1, partition=partition)

    def test_C_leaf_is_one_scene_and_historical_monolith_is_ledger_only(self):
        _, _, v3, ref1, ref2, ref3 = self.versions()
        catalog = [ref2, ref3]
        scene = "FloorPlan201"
        config = {"collect_only": False, "pilot_scenes": [scene],
                  "shard": {"partition": "legacy", "scene": scene, "manifest": ref3}}
        merger.validate_leaf_scope(config, [scene], "legacy", v3, catalog, ref1)
        old = {"collect_only": True, "pilot_scenes": v3["partitions"]["legacy"]["scenes"]}
        merger.validate_leaf_scope(old, old["pilot_scenes"], "legacy", v3, catalog, ref1)
        self.assertEqual(merger.measurement_policy(old, {}), "legacy_collect_only")
        for bad in ({**config, "pilot_scenes": old["pilot_scenes"]},
                    {**config, "shard": {**config["shard"], "manifest": ref2}},
                    {**old, "collect_only": False}):
            with self.assertRaises(ValueError):
                merger.validate_leaf_scope(bad, bad["pilot_scenes"], "legacy", v3, catalog, ref1)

    def test_main_merges_v2_A_B_v3_C_and_explicit_old_legacy_link(self):
        # Exercise actual expansion/ownership/leaf control flow. Only the external
        # provenance reader is mocked; scientific rows are absent, never accepted.
        _, v2, v3, _, ref2, ref3 = self.versions()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "results").mkdir()
            documents = {}
            def add(name, config, metrics, sources=()):
                relative = f"results/{name}.json"
                reference = {"file": relative, "sha256": name, "exit_code": 0}
                documents[str(root / relative)] = ({"config": config, "metrics": metrics,
                                                   "sources": list(sources)}, reference)
                return reference
            old = add("old", {"collect_only": True, "seed": 0,
                               "pilot_scenes": v3["partitions"]["legacy"]["scenes"]},
                      {"kind": "t01_pair_generator", "attempts": 0, "status": "synthetic_empty"})
            parents = []
            for partition, manifest, ref in (("a", v2, ref2), ("b", v2, ref2), ("legacy", v3, ref3)):
                scene = manifest["partitions"][partition]["scenes"][0]
                child = add(partition + "_child", {"collect_only": False, "acceptance_policy": POLICY,
                    "seed": 0, "pilot_scenes": [scene], "shard": {"partition": partition,
                    "scene": scene, "manifest": ref, "pbs_jobid": "fixture"}},
                    {"kind": "t01_scene_collect", "attempts": 0, "status": "synthetic_empty"})
                parent = add(partition, {"partition": partition, "manifest": ref, "collect_only": False,
                    "acceptance_policy": POLICY["id"], "seed": 0, "pbs_jobid": "fixture",
                    "pilot_scenes": manifest["partitions"][partition]["scenes"]},
                    {"kind": "t01_parallel_collect", "status": "synthetic_empty",
                     "scene_runs": [{"scene": scene, "result_path": child["file"]}]}, [child])
                parents.append(parent["file"])
            argv = ["merge", "--name", "pilot_pairs_C_fixture", "--source", old["file"], *parents,
                    "--retry-of", f"{parents[-1]}={old['file']}", "--scratch", str(root / "scratch")]
            with patch.object(merger, "ROOT", root), patch.object(merger, "audited_result", side_effect=lambda path: documents[str(path)]), \
                 patch.object(merger, "write_result"), patch.object(sys, "argv", argv):
                self.assertEqual(merger.main(), 0)
            result = json.loads((root / "results/pilot_pairs_C_fixture.json").read_text())
            self.assertEqual(result["metrics"]["aggregation_status"], "completed")
            self.assertFalse(result["metrics"]["acceptance_passed"])
            self.assertEqual(len(result["metrics"]["source_runs"]), 4)
            self.assertEqual(result["metrics"]["source_runs"][0]["measurement_policy"], "legacy_collect_only")


if __name__ == "__main__":
    unittest.main()

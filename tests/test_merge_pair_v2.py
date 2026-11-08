"""Synthetic CPU checks only; temporary fixtures are not research results."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from src.tools.merge_pair_v2 import (KINDS, balanced_selection, durations,
    journal_reference, load_observation, producer_arguments, proposal_identity,
    publication_eligible, reference_path, scene_number, source_eligibility,
    verify_preflight, verify_source_identity)


def source(exit_code=0, status="completed_search", cleanup=None):
    return {"_meta": {"exit_code": exit_code, "end": "synthetic-finished"},
            "metrics": {"kind": "pair_v2_shard", "status": status,
                        "cleanup_errors": [] if cleanup is None else cleanup},
            "config": {"seed": 0, "resolution": 255}}


def candidate(kind, index, scene="FloorPlan1", source_file="results/synthetic.json"):
    return {"kind": kind, "journal_index": index, "scene": scene, "source_file": source_file,
            "proposal_identity": kind + "_" + str(index), "target_w0": "a", "target_w1": "b"}


def proposal():
    point = lambda x: {"x": x, "y": 1.0, "z": 0.0}
    return {"scene": "FloorPlan1", "identities": {"candidate_names": ["a", "b"], "anchor_name": "x"},
            "proposal": {"w0": {"spawn": {"selected_position": point(0), "receptacle_name": "table"}},
                         "w1": {"spawn": {"selected_position": point(1), "receptacle_name": "table"}}},
            "view": {"position": point(2), "yaw": 45, "horizon": 30}}


def identity_fixture():
    device = {"logical_index": 0, "gpu_index": 2, "gpu_uuid": "GPU-synthetic",
              "gpu_name": "NVIDIA H200", "vulkan_device_index": 3, "mig_mode": "Disabled"}
    allocation = {"pbs_jobid": "synthetic.hopper", "devices": [device],
                  "cuda_visible_devices": "GPU-synthetic", "cpu_affinity": list(range(12))}
    runtime = {key: value for key, value in device.items() if key != "logical_index"}
    runtime.update(pbs_jobid="synthetic.hopper", cuda_visible_devices="GPU-synthetic", cpu_affinity=list(range(12)))
    data = source()
    data["_meta"]["name"] = "synthetic"
    data["config"].update(name="synthetic", scene="FloorPlan1", raw_evidence_dir="/synthetic/raw", runtime=runtime)
    data["config"]["shard"] = {"name": "synthetic", "scene": "FloorPlan1", "seed": 0, "resolution": 255,
        "gpu_identity": device, "logical_index": 0, "allocation": allocation, "pbs_jobid": "synthetic.hopper",
        "parent_cuda_visible_devices": "GPU-synthetic", "child_cuda_visible_devices": "GPU-synthetic",
        "node": "synthetic-node", "cpu_affinity": list(range(12)), "raw_evidence_dir": "/synthetic/raw"}
    capture = {"valid_shape": True, "action_success": True, "black": False,
               "shape": [255, 255, 3], "segmentation": {"valid": True}}
    gate = {"_meta": {"exit_code": 0, "end": "synthetic-finished"},
        "config": {"allocation": allocation, "pbs_jobid": "synthetic.hopper", "node": "synthetic-node",
                   "resolution": 255, "segmentation": True},
        "metrics": {"passed": True, "cleanup_errors": [], "errors": [], "expected_gpu_count": 1,
                    "completed_gpu_count": 1, "devices": [{"worker_error": None,
                        "observation": {"logical_index": 0, "gpu_identity": device, "node": "synthetic-node",
                                        "valid": True, "captures": [copy.deepcopy(capture), copy.deepcopy(capture)]}}]}}
    return copy.deepcopy((data, gate, runtime, allocation))


class MergePairV2Tests(unittest.TestCase):
    def test_exact_python_module_producer_with_optional_unbuffered_flag(self):
        for flag in ([], ["-u"]):
            command = ["/synthetic/bin/python3.11", *flag, "-m", "src.envs.pair_v2_shard", "--name", "synthetic"]
            self.assertEqual(producer_arguments(command, "src.envs.pair_v2_shard"), ["--name", "synthetic"])
        for command in (["python", "-m", "other", "src.envs.pair_v2_shard"],
                        ["sh", "-m", "src.envs.pair_v2_shard"],
                        ["python", "-c", "pass", "-m", "src.envs.pair_v2_shard"]):
            with self.assertRaises(ValueError):
                producer_arguments(command, "src.envs.pair_v2_shard")

    def test_source_identity_matches_actual_runtime_and_gate(self):
        data, gate, runtime, allocation = identity_fixture()
        verify_source_identity(data, gate, logged_runtime=runtime, logged_allocation=allocation)

    def test_wrong_job_uuid_cpu_or_changed_runtime_log_is_rejected(self):
        for field, value in (("pbs_jobid", "other.hopper"), ("gpu_uuid", "GPU-other"),
                             ("cpu_affinity", list(range(1, 13)))):
            data, gate, runtime, allocation = identity_fixture()
            data["config"]["runtime"][field] = value
            with self.assertRaises(ValueError):
                verify_source_identity(data, gate, logged_runtime=runtime, logged_allocation=allocation)
        data, gate, runtime, allocation = identity_fixture()
        data["config"]["shard"]["pbs_jobid"] = "other.hopper"
        with self.assertRaises(ValueError):
            verify_source_identity(data, gate, logged_runtime=runtime, logged_allocation=allocation)

    def test_failed_before_generation_keeps_missing_attempts_unknown(self):
        data = source(1, "failed_before_generation")
        data["config"]["attempt_journal"] = "data/world_pairs/v2/synthetic.jsonl"
        self.assertIsNone(journal_reference(data, source_eligibility(data)))
        self.assertNotIn("attempts", data["metrics"])
        data["metrics"]["attempts"] = 1
        with self.assertRaises(ValueError):
            journal_reference(data, source_eligibility(data))

    def test_ledger_only_failure_does_not_veto_other_complete_pairs(self):
        failed = source_eligibility(source(1, "worker_failed"))
        self.assertFalse(failed["eligible"])
        counts = {kind: 20 for kind in KINDS}
        self.assertTrue(publication_eligible(counts, [{"source": "successful"}, failed], [], []))
        self.assertFalse(publication_eligible(counts, [{"source": "successful"}, failed], ["SHA mismatch"], []))

    def test_ended_failed_source_is_explicit_partial_and_can_contribute(self):
        result = source_eligibility(source(1, "incomplete"))
        self.assertTrue(result["eligible"])
        self.assertTrue(result["partial_source"])
        self.assertEqual(result["exit_code"], 1)

    def test_running_or_unknown_cleanup_source_never_contributes(self):
        for status in ("running", "worker_failed", "failed_before_generation"):
            self.assertFalse(source_eligibility(source(1, status))["eligible"])
        value = source()
        value["metrics"].pop("cleanup_errors")
        self.assertFalse(source_eligibility(value)["eligible"])
        self.assertFalse(source_eligibility(source(cleanup=["ownership unknown"]))["eligible"])

    def test_missing_formal_finish_or_wrong_protocol_is_rejected(self):
        for field, replacement in (("exit_code", None), ("end", None)):
            value = source()
            value["_meta"][field] = replacement
            self.assertFalse(source_eligibility(value)["eligible"])
        value = source()
        value["config"]["resolution"] = 224
        self.assertFalse(source_eligibility(value)["eligible"])

    def test_first_twenty_per_class_retains_surplus_and_numeric_order(self):
        rows = [candidate(kind, index, "FloorPlan2" if index < 21 else "FloorPlan10")
                for kind in KINDS for index in range(22)]
        selected, surplus, duplicate = balanced_selection(list(reversed(rows)))
        self.assertEqual(len(selected), 80)
        self.assertEqual(len(surplus), 8)
        self.assertEqual(duplicate, [])
        for kind in KINDS:
            self.assertEqual([r["journal_index"] for r in selected if r["kind"] == kind], list(range(20)))

    def test_duplicate_proposal_is_not_counted_twice(self):
        first = candidate("look_flip", 0)
        duplicate = {**first, "source_file": "results/retry.json"}
        selected, surplus, duplicates = balanced_selection([first, duplicate])
        self.assertEqual(len(selected), 1)
        self.assertEqual(len(duplicates), 1)
        self.assertEqual(surplus, [])

    def test_conflicting_duplicate_is_not_resolved_by_best_outcome(self):
        first = candidate("look_flip", 0)
        second = {**first, "kind": "direct_flip"}
        with self.assertRaises(ValueError):
            balanced_selection([first, second])

    def test_missing_duration_is_unknown_not_zero(self):
        self.assertEqual(durations([None, float("nan")])["mean_seconds"], None)
        result = durations([None, 0.0, 2.0, float("inf")])
        self.assertEqual(result, {"observed_count": 2, "missing_count": 2,
                                  "mean_seconds": 1.0, "total_observed_seconds": 2.0})

    def test_source_paths_do_not_change_proposal_identity(self):
        a = proposal()
        b = copy.deepcopy(a)
        b["proposal"]["w0"]["spawn"]["query_reference"] = {"file": "different/run"}
        self.assertEqual(proposal_identity(a), proposal_identity(b))
        b["proposal"]["w1"]["spawn"]["selected_position"]["x"] = 3
        self.assertNotEqual(proposal_identity(a), proposal_identity(b))

    def test_invalid_scene_fails(self):
        for name in ("FloorPlan", "FloorPlan0", "../FloorPlan1", None):
            with self.assertRaises(ValueError):
                scene_number(name)
        self.assertLess(scene_number("FloorPlan2"), scene_number("FloorPlan10"))

    def test_preflight_uses_final_rest_without_extra_transient_gate(self):
        original = [{"name": "box", "moveable": True, "position": {"x": 0, "y": 0, "z": 0}}]
        final = copy.deepcopy(original)
        final[0]["position"]["x"] = 0.01
        value = {"passed": True, "snapshot": {"objects": final, "native_is_scene_at_rest": True},
            "diagnostics": {"source_snapshot": {"objects": original},
                "settling": {"settled": True, "observations": [
                    {"native_is_scene_at_rest": False, "synthetic_transient_displacement_m": 0.03},
                    {"native_is_scene_at_rest": True}]}}}
        self.assertTrue(verify_preflight(value)["passed"])
        final[0]["position"]["x"] = 0.010001
        with self.assertRaises(ValueError):
            verify_preflight(value)

    def test_artifact_hash_and_directory_are_both_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inner = root / "owned"
            inner.mkdir()
            path = inner / "evidence.json"
            path.write_text('{"synthetic":true}')
            reference = {"file": "owned/evidence.json", "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            self.assertEqual(reference_path(reference, root=root, within=inner), path)
            with self.assertRaises(ValueError):
                reference_path({**reference, "sha256": "0" * 64}, root=root, within=inner)
            with self.assertRaises(ValueError):
                reference_path(reference, root=root, within=root / "other")

    def test_portable_observation_uses_real_files_mapping(self):
        import numpy as np
        from PIL import Image
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = {"metadata": {"objects": []}, "marks": [], "mark_labels_by_name": {},
                      "input_text": "synthetic", "history": [], "files": {}}
            portable = {}
            for field in ("rgb", "marked", "masks"):
                path = root / (field + (".npz" if field == "masks" else ".png"))
                if field == "masks":
                    np.savez_compressed(path, names=np.asarray(["x"]), masks=np.ones((1, 255, 255), dtype=bool))
                else:
                    Image.fromarray(np.full((255, 255, 3), 17, dtype=np.uint8)).save(path)
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                record["files"][field] = {"file": path.name, "sha256": digest}
                portable[path] = digest
            observed = load_observation(record, portable, root, root=root)
            self.assertEqual(observed["rgb"].shape, (255, 255, 3))
            self.assertTrue(observed["masks_by_name"]["x"].all())
            del portable[root / "masks.npz"]
            with self.assertRaises(ValueError):
                load_observation(record, portable, root, root=root)


if __name__ == "__main__":
    unittest.main()

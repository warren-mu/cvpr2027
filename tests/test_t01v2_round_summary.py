"""Synthetic CPU contracts; no scene, actual results, or simulator is read."""
import copy
import unittest

from src.tools.t01v2_round_summary import KINDS, summarize_source


def fixture():
    name = "pilot_pairs_v2_synthetic_floorplan1"
    device = {"logical_index": 0, "gpu_uuid": "GPU-synthetic"}
    allocation = {"pbs_jobid": "synthetic.hopper", "devices": [device],
                  "cuda_visible_devices": "GPU-synthetic"}
    shard = {"name": name, "scene": "FloorPlan1", "seed": 0, "resolution": 255,
        "pbs_jobid": "synthetic.hopper", "logical_index": 0, "gpu_identity": device,
        "allocation": allocation, "child_cuda_visible_devices": "GPU-synthetic",
        "parent_cuda_visible_devices": "GPU-synthetic"}
    config = {key: shard[key] for key in ("name", "scene", "seed", "resolution")}
    config.update(shard=shard, runtime={"pbs_jobid": "synthetic.hopper", "gpu_uuid": "GPU-synthetic",
                                      "cuda_visible_devices": "GPU-synthetic"})
    data = {"config": config, "_meta": {"name": name, "exit_code": 0, "end": "synthetic-ended",
        "command": ["/synthetic/bin/python3", "-u", "-m", "src.envs.pair_v2_shard",
                    "--name", name, "--scene", "FloorPlan1"]},
        "metrics": {"kind": "pair_v2_shard", "status": "completed_search", "cleanup_errors": [],
                    "attempts": 20, "passed_by_class": {kind: 5 for kind in KINDS},
                    "discard_reason_counts": {}}}
    reference = {"file": "results/" + name + ".json", "audit": {"fail": [], "warn": []}}
    return copy.deepcopy((data, reference))


class RoundSummaryTests(unittest.TestCase):
    def test_five_boundary_and_failed_partial_counts_remain_reported(self):
        data, ref = fixture()
        self.assertFalse(summarize_source(data, ref)["needs_clone_next_round"])
        data["metrics"]["passed_by_class"]["look_flip"] = 4
        data["metrics"].update(status="incomplete", cleanup_errors=["synthetic cleanup error"])
        data["_meta"]["exit_code"] = 1
        row = summarize_source(data, ref)
        self.assertTrue(row["needs_clone_next_round"])
        self.assertEqual(row["below_threshold_classes"], ["look_flip"])
        self.assertEqual(row["cleanup_errors"], ["synthetic cleanup error"])
        self.assertEqual(row["exit_code"], 1)

    def test_missing_counts_and_attempts_are_unknown_not_zero(self):
        for missing in (None, {}, {"look_flip": 0}, {**dict.fromkeys(KINDS, 5), "direct_flip": None}):
            data, ref = fixture()
            data["metrics"]["passed_by_class"] = missing
            data["metrics"].pop("attempts")
            row = summarize_source(data, ref)
            self.assertIsNone(row["needs_clone_next_round"])
            self.assertIsNone(row["reported_attempts"])
            self.assertEqual(row["reported_passed_by_class"], missing)
            self.assertTrue(row["unknown_count_classes"])

    def test_negative_boolean_or_noninteger_counts_are_rejected(self):
        for value in (-1, True, 4.0, "4"):
            data, ref = fixture()
            data["metrics"]["passed_by_class"]["look_flip"] = value
            with self.assertRaises(ValueError):
                summarize_source(data, ref)

    def test_wrong_command_or_scene_or_runtime_identity_is_rejected(self):
        for mode in ("module_tail", "scene", "runtime_uuid", "job", "unfinished"):
            data, ref = fixture()
            if mode == "module_tail":
                data["_meta"]["command"][3] = "other.module"
                data["_meta"]["command"].append("src.envs.pair_v2_shard")
            elif mode == "scene": data["config"]["shard"]["scene"] = "FloorPlan2"
            elif mode == "runtime_uuid": data["config"]["runtime"]["gpu_uuid"] = "GPU-other"
            elif mode == "job": data["config"]["shard"]["pbs_jobid"] = "other.hopper"
            else: data["_meta"]["end"] = None
            with self.assertRaises(ValueError):
                summarize_source(data, ref)

    def test_ended_before_runtime_preserves_declared_identity_and_unknown_counts(self):
        data, ref = fixture()
        data["config"].pop("runtime")
        data["metrics"] = {"kind": "pair_v2_shard", "status": "failed_before_generation",
                           "cleanup_errors": [], "black_frame_detected": False}
        data["_meta"]["exit_code"] = 1
        row = summarize_source(data, ref)
        self.assertFalse(row["runtime_identity_observed"])
        self.assertIsNone(row["needs_clone_next_round"])
        self.assertIsNone(row["reported_passed_by_class"])


if __name__ == "__main__":
    unittest.main()

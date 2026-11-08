"""CPU checks for GPU identity selection and budget overcommit prevention."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from src.envs.thor_allocation import make_controller_class, match_devices

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("submit_phase", ROOT / "scripts/pbs/submit_phase.py")
submit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(submit)

A = "GPU-11111111-1111-1111-1111-111111111111"
B = "GPU-22222222-2222-2222-2222-222222222222"
ROWS = [["3", A, "NVIDIA H200"], ["7", B, "NVIDIA H200"]]


class AllocationSelectionTests(unittest.TestCase):
    def test_uuid_order_is_pbs_order_not_nvml_order(self):
        devices = match_devices(f"{B},{A}", ROWS, 2)
        self.assertEqual([x["gpu_index"] for x in devices], [7, 3])
        self.assertEqual([x["logical_index"] for x in devices], [0, 1])

    def test_numeric_mask_uses_physical_identity(self):
        self.assertEqual(match_devices("7", ROWS, 1)[0]["gpu_uuid"], B)

    def test_no_duplicate_or_unassigned_gpu(self):
        for mask in (f"{A},{A}", "3,9", f"{A},7", "MIG-123", "GPU-1111"):
            with self.subTest(mask=mask), self.assertRaises(RuntimeError):
                match_devices(mask, ROWS, 2)

    def test_hardware_and_count_must_match(self):
        with self.assertRaises(RuntimeError):
            match_devices("3", ROWS, 2)
        with self.assertRaises(RuntimeError):
            match_devices("3", [["3", A, "NVIDIA A100"]], 1)

    def test_controller_caches_are_private_and_gpu_is_selected_by_uuid(self):
        package, build, controller = (types.ModuleType(name) for name in
                                      ("ai2thor", "ai2thor.build", "ai2thor.controller"))
        build.base_url = "http://example.invalid/build"
        package.build = build

        class FakeController:
            def __init__(self, **kwargs):
                self.received_gpu_device = kwargs["gpu_device"]

        controller.Controller = FakeController
        modules = {"ai2thor": package, "ai2thor.build": build, "ai2thor.controller": controller}
        with tempfile.TemporaryDirectory() as folder, patch.dict("sys.modules", modules):
            runtime = {"pbs_jobid": "fixture.hopper-m-02", "cuda_visible_devices": f"{B},{A}",
                       "scratch": folder, "log_dir": str(Path(folder) / "logs"),
                       "devices": [dict(logical_index=0, gpu_index=7, gpu_uuid=B, vulkan_device_index=2),
                                   dict(logical_index=1, gpu_index=3, gpu_uuid=A, vulkan_device_index=5)]}
            with patch.dict(os.environ, {"PBS_JOBID": runtime["pbs_jobid"],
                                         "CUDA_VISIBLE_DEVICES": runtime["cuda_visible_devices"]}):
                original_home = os.environ.get("HOME")
                cls = make_controller_class(runtime)
                first, second = cls(gpu_device=0), cls(gpu_device=1)
                self.assertEqual((first.received_gpu_device, second.received_gpu_device), (7, 3))
                self.assertNotEqual(first.base_dir, second.base_dir)
                self.assertEqual(first.releases_dir, second.releases_dir)
                self.assertEqual(json.loads((Path(first.base_dir) / "cuda-vulkan-mapping.json").read_text()),
                                 {"7": 2, "3": 5})
                self.assertEqual(os.environ.get("HOME"), original_home)
                self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], runtime["cuda_visible_devices"])
                with self.assertRaises(ValueError):
                    cls(gpu_device=2)
                with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": A}), self.assertRaises(RuntimeError):
                    cls(gpu_device=0)


class BudgetTests(unittest.TestCase):
    def state(self, t00b=0, t01=0, names=()):
        return {"accounted_or_reserved_by_task": {"T00b": t00b, "T01": t01},
                "accounted_or_reserved_total": t00b + t01,
                "submissions": [{"name": name} for name in names]}

    def test_two_gpu_request_reserves_both_cards(self):
        submit.validate_request("T00b", 2, 10800, "t00b_training", self.state(t00b=2))
        with self.assertRaises(ValueError):
            submit.validate_request("T00b", 2, 10800, "t00b_training", self.state(t00b=2.01))

    def test_task_budget_still_applies_with_global_room(self):
        with self.assertRaises(ValueError):
            submit.validate_request("T01", 1, 3600, "pilot_pairs", self.state(t01=3.01))

    def test_total_budget_also_enforced(self):
        state = self.state(t00b=1, t01=0)
        state["accounted_or_reserved_total"] = 11.1
        with self.assertRaises(ValueError):
            submit.validate_request("T00b", 1, 3600, "t00b_retry", state)

    def test_second_submission_cannot_share_output_name(self):
        with self.assertRaises(ValueError):
            submit.validate_request("T01", 1, 60, "pilot_pairs", self.state(names=["pilot_pairs"]))

    def test_t01_stays_single_gpu_and_t00b_within_three_hours(self):
        for task, gpus, seconds, name in (("T01", 2, 60, "pilot_pairs"),
                                            ("T00b", 2, 10801, "t00b_training"),
                                            ("T01", 1, 60, "t00b_wrong_task")):
            with self.subTest(task=task, name=name), self.assertRaises(ValueError):
                submit.validate_request(task, gpus, seconds, name, self.state())

    def test_lost_submission_acknowledgement_does_not_release_budget(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            item = root / "logs/pbs/uncertain_submission"
            item.mkdir(parents=True)
            payload = {"budget_batch": submit.BATCH, "task_id": "T00b", "name": "t00b_uncertain",
                       "requested_gpu_count": 2, "requested_wall_seconds": 10800, "dry_run": False}
            (item / "payload.json").write_text(json.dumps(payload))
            (item / "submission.json").write_text(json.dumps({"returncode": 1, "stdout": "",
                                                               "stderr": "connection lost"}))
            result = submit.budget_state(root)
            self.assertEqual(result["accounted_or_reserved_by_task"]["T00b"], 6.0)
            self.assertEqual(result["submissions"][0]["basis"], "full_request_reservation")

    def test_only_substantiated_final_accounting_replaces_reservation(self):
        fields = {"job_state": "F", "resources_used.walltime": "01:30:00", "Resource_List.ngpus": "2"}
        self.assertEqual(submit.recorded_final_hours(fields, 3.0, "fixture"), 3.0)
        self.assertIsNone(submit.recorded_final_hours({**fields, "job_state": "R"}, 3.0, "fixture"))
        self.assertIsNone(submit.recorded_final_hours({"job_state": "F"}, 0.0, "fixture"))
        self.assertEqual(submit.recorded_final_hours({"job_state": "F", "run_count": "0"}, 0.0, "fixture"), 0.0)
        with self.assertRaises(RuntimeError):
            submit.recorded_final_hours(fields, 1.0, "fixture")


if __name__ == "__main__":
    unittest.main()

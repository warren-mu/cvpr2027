"""Synthetic CPU fixtures only; root runs after commit through run_logged."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.tools import phase_watch as watch
from src.tools.t01v2_parallel import (scene_plan, shard_configuration, submission_identity,
                                     validate_mask, verify_shard_result)
from src.tools.t01_partition import cpu_shards

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("v2_submit", ROOT / "scripts/pbs/submit_phase.py")
submit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(submit)
JOB = "123.hopper-m-02"
UUIDS = ["GPU-11111111-1111-1111-1111-111111111111", "GPU-22222222-2222-2222-2222-222222222222"]


def policy():
    return {"rates": {q: 1.0 for q in ("smallx", "mediumx", "largex")}}


def state(v2=0, total=0):
    return {"submissions": [], "charge_policy": policy(), "accounted_or_reserved_by_task": {"T00b": 0, "T01": 0},
            "accounted_or_reserved_total": total, "accounted_or_reserved_by_subbudget": {"T01v2": v2}}


def payload(name="pilot_pairs_v2_fixture", job=JOB):
    return {"name": name, "task_id": "T01", "budget_group": "T01v2", "budget_batch": submit.BATCH,
            "requested_gpu_count": 2, "requested_wall_seconds": 1800, "requested_queue_charge_rate": 1,
            "requested_host": None, "queue": "autox", "dry_run": False,
            "command": ["bash", "experiments/t01v2_pairs.sh", "--name", name, "--gpus", "2", "--scenes", "FloorPlan1", "FloorPlan2"]}


def write_submission(root, directory, data, receipt=JOB):
    item = root / "logs/pbs" / directory
    item.mkdir(parents=True)
    (item / "payload.json").write_text(json.dumps(data))
    (item / "submission.json").write_text(json.dumps({"stdout": receipt or "", "returncode": 0 if receipt else 1}))
    return item


class V2BudgetTests(unittest.TestCase):
    def test_subbudget_is_additional_to_global_and_rate_aware(self):
        submit.validate_request("T01", 2, 1800, "pilot_pairs_v2_next", state(15, 59))
        for value in (state(15.01, 0), state(0, 59.01)):
            with self.assertRaises(ValueError):
                submit.validate_request("T01", 2, 1800, "pilot_pairs_v2_next", value)
        doubled = state(15)
        doubled["charge_policy"]["rates"]["smallx"] = 2
        with self.assertRaises(ValueError):
            submit.validate_request("T01", 2, 1800, "pilot_pairs_v2_next", doubled)
        missing = state()
        del missing["accounted_or_reserved_by_subbudget"]
        with self.assertRaises(ValueError):
            submit.validate_request("T01", 2, 1800, "pilot_pairs_v2_next", missing)

    def test_group_cannot_be_omitted_by_name_or_explicit_launcher(self):
        command = payload()["command"]
        self.assertEqual(submit.budget_group("T01", "pilot_pairs_v2_fixture", command), "T01v2")
        self.assertEqual(submit.budget_group("T01", "pilot_pairs_v2_fixture", []), "T01v2")
        self.assertIsNone(submit.budget_group("T00b", "t00b_scale4", []))
        for task, name, declared in (("T00b", "pilot_pairs_v2_fixture", None),
                                     ("T01", "pilot_pairs_other", None),
                                     ("T01", "pilot_pairs_v2_fixture", "other")):
            with self.assertRaises(ValueError):
                submit.budget_group(task, name, command, declared)

    def test_final_cost_pending_full_request_unknown_ack_all_count_once(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_submission(root, "final", payload(), JOB)
            write_submission(root, "pending", payload("pilot_pairs_v2_pending"), "124.hopper-m-02")
            uncertain = payload("pilot_pairs_v2_uncertain")
            uncertain["requested_queue_charge_rate"] = 2
            write_submission(root, "uncertain", uncertain, None)
            old = payload("pilot_pairs_v1_other")
            old.update(budget_group=None, command=["bash", "experiments/t01_parallel_pairs.sh"])
            write_submission(root, "v1", old, "125.hopper-m-02")
            final = {JOB: {"gpu_hours": .25, "queue_charge_rate": 2, "accounting_result": "results/final.json"}}
            with patch.object(submit, "_final_accounting", return_value=final):
                ledger = submit.budget_state(root, policy(), {"124.hopper-m-02": 1, "125.hopper-m-02": 1})
            self.assertEqual(ledger["accounted_or_reserved_by_subbudget"]["T01v2"], 3.5)
            self.assertEqual(ledger["accounted_or_reserved_by_task"]["T01"], 4.5)
            self.assertEqual(ledger["accounted_or_reserved_total"], 4.5)
            self.assertEqual(ledger["submissions"][0]["basis"], "final_accounting")

    def test_runtime_requires_one_same_job_unpinned_reserved_submission(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = payload()
            directory = write_submission(root, "original", data)
            self.assertEqual(submission_identity(root, data["name"], JOB, 2, 1700)["budget_group"], "T01v2")
            for key, value in (("budget_group", None), ("requested_host", "hopper-29"),
                               ("requested_gpu_count", 4), ("requested_wall_seconds", 1700)):
                bad = {**data, key: value}
                (directory / "payload.json").write_text(json.dumps(bad))
                with self.subTest(key=key), self.assertRaises(ValueError):
                    submission_identity(root, data["name"], JOB, 2, 1700)
            (directory / "payload.json").write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                submission_identity(root, data["name"], "999.hopper-m-02", 2, 1700)
            write_submission(root, "duplicate", data)
            with self.assertRaises(ValueError):
                submission_identity(root, data["name"], JOB, 2, 1700)


class V2ShardTests(unittest.TestCase):
    def test_fixed_nonoverlapping_scene_and_cpu_assignment(self):
        self.assertEqual(scene_plan(["FloorPlan1", "FloorPlan2", "FloorPlan3", "FloorPlan4"], 2),
                         [["FloorPlan1", "FloorPlan3"], ["FloorPlan2", "FloorPlan4"]])
        cpus = cpu_shards(range(10, 34), 2)
        self.assertEqual(cpus, [list(range(10, 22)), list(range(22, 34))])
        for scenes, count in ((["FloorPlan1", "FloorPlan1"], 2), (["FloorPlan1"], 2),
                               (["FloorPlan31", "FloorPlan2"], 2), (["FloorPlan1"], True)):
            with self.assertRaises(ValueError):
                scene_plan(scenes, count)
        with self.assertRaises(ValueError):
            cpu_shards(range(23), 2)

    def test_mask_is_original_uuid_order_only(self):
        self.assertEqual(validate_mask(",".join(reversed(UUIDS)), 2), list(reversed(UUIDS)))
        for mask in ("0,1", ",".join([UUIDS[0]] * 2), UUIDS[0], ",".join(UUIDS) + ",GPU-other"):
            with self.assertRaises(ValueError):
                validate_mask(mask, 2)

    def test_child_identity_and_cleanup_are_required_even_after_exit_zero(self):
        devices = [{"logical_index": index, "gpu_uuid": value} for index, value in enumerate(UUIDS)]
        gate = {"config": {"node": "hopper-11", "allocation": {"pbs_jobid": JOB, "devices": devices,
                  "cuda_visible_devices": ",".join(UUIDS)}}}
        expected = shard_configuration("pilot_pairs_v2_fixture_g1", "FloorPlan2", 1, list(range(12, 24)),
            gate, {"sha256": "gatehash"}, "results/gate.json", ["FloorPlan1", "FloorPlan2"], Path("/scratch/fixture"))
        record = {"exit_code": 0}
        result = {"_meta": {"name": expected["name"], "exit_code": 0},
                  "config": {"shard": expected}, "metrics": {"kind": "pair_v2_shard", "cleanup_errors": []}}
        verify_shard_result(record, result, expected)
        for key, value in (("scene", "FloorPlan1"), ("child_cuda_visible_devices", UUIDS[0]),
                           ("cpu_affinity", list(range(12))), ("pbs_jobid", "999.hopper-m-02")):
            bad = copy.deepcopy(result)
            bad["config"]["shard"][key] = value
            with self.assertRaises(ValueError):
                verify_shard_result(record, bad, expected)
        bad = copy.deepcopy(result)
        del bad["metrics"]["cleanup_errors"]
        with self.assertRaises(ValueError):
            verify_shard_result(record, bad, expected)

    def test_watch_v2_entrypoint_and_only_v2_module(self):
        command = payload()["command"]
        self.assertEqual(watch.entrypoint(command, "pilot_pairs_v2_fixture", ROOT), "t01v2")
        self.assertTrue(watch.authorized_shard_command(["/scratch/python3", "-u", "-m", "src.envs.pair_v2_shard"], v2=True))
        self.assertFalse(watch.authorized_shard_command(["python3", "-m", "src.envs.pair_shard_approved"], v2=True))
        self.assertFalse(watch.authorized_shard_command(["python3", "-m", "src.envs.pair_v2_shard"]))
        self.assertFalse(watch.authorized_shard_command(["python3", "-c", "x", "src.envs.pair_v2_shard"], v2=True))


def black_fixture(child_failure=False):
    data = payload()
    tag = JOB.replace(".", "_").replace("-", "_")
    gate_name = data["name"] + "_j" + tag + "_gate"
    devices = [{"logical_index": index, "gpu_uuid": value} for index, value in enumerate(UUIDS)]
    code = 0 if child_failure else 86
    gate = {"_meta": {"name": gate_name, "exit_code": code,
                      "command": ["python3", "-u", "-m", "src.tools.render_gate", "--name", gate_name]},
            "config": {"pbs_jobid": JOB, "node": "hopper-11", "resolution": 255, "segmentation": True,
                       "allocation": {"pbs_jobid": JOB, "devices": devices, "cuda_visible_devices": ",".join(UUIDS)}},
            "metrics": {"exit_code": code, "passed": child_failure, "black_frame_detected": not child_failure,
                        "expected_gpu_count": 2, "completed_gpu_count": 2, "errors": [], "cleanup_errors": [],
                        "devices": [{"observation": {"gpu_identity": device, "captures": [
                            {"valid_shape": True, "action_success": True, "valid": child_failure,
                             "black": not child_failure, "nonzero_rgb_values": 1 if child_failure else 0}
                            for _ in range(2)]}} for device in devices]}}
    outer = {"_meta": {"name": data["name"], "command": data["command"], "exit_code": 86},
             "config": {"pbs_jobid": JOB}, "metrics": {"kind": "pair_v2_parallel", "cleanup_errors": []}}
    fields = {"project": "CFP04-SF-125", "job_state": "F", "Exit_status": "86", "exec_host": "hopper-11/0*24"}
    record = {"run_name": data["name"] + "_scene", "exit_code": 86, "scene": "FloorPlan1",
              "gpu_uuid": UUIDS[0], "logical_index": 0}
    child = {"_meta": {"name": record["run_name"], "exit_code": 86,
                       "command": ["/scratch/python3", "-u", "-m", "src.envs.pair_v2_shard"]},
             "config": {"shard": {"pbs_jobid": JOB, "scene": "FloorPlan1", "node": "hopper-11",
                "gpu_identity": devices[0], "child_cuda_visible_devices": UUIDS[0],
                "parent_cuda_visible_devices": ",".join(UUIDS)}},
             "metrics": {"kind": "pair_v2_shard", "cleanup_errors": [], "black_frame_detected": True,
                         "source_black_frame_detected": True, "capture_black_frame_detected": False,
                         "black_frame_stop": {"scene": "FloorPlan1", "stage": "source", "measurement": {
                             "shape": [255, 255, 3], "dtype": "uint8", "min": 0, "max": 0, "nonzero_values": 0}}}}
    return data, fields, outer, gate, gate_name, child, record


class V2RetryTests(unittest.TestCase):
    def test_initial_gate_requires_v2_budget_and_explicit_cleanup(self):
        data, fields, outer, gate, name, _, _ = black_fixture()
        self.assertEqual(watch.validate_black_retry(data, JOB, fields, outer, gate, name, "t01v2"),
                         "render_gate_black_capture")
        for change in (lambda p, o, g: p.update(budget_group=None),
                       lambda p, o, g: p.update(requested_host="hopper-29"),
                       lambda p, o, g: o["metrics"].pop("cleanup_errors"),
                       lambda p, o, g: g["config"].update(resolution=224),
                       lambda p, o, g: g["config"].update(node="hopper-12")):
            p, o, g = copy.deepcopy((data, outer, gate))
            change(p, o, g)
            with self.assertRaises(ValueError):
                watch.validate_black_retry(p, JOB, fields, o, g, name, "t01v2")

    def test_v2_capture_does_not_accept_v1_shape_module_or_unknown_cleanup(self):
        data, fields, outer, gate, name, child, record = black_fixture(True)
        self.assertEqual(watch.validate_black_retry(data, JOB, fields, outer, gate, name, "t01v2", child, record),
                         "t01v2_scene_black_capture")
        for change in (lambda c: c["metrics"].pop("cleanup_errors"),
                       lambda c: c["config"]["shard"].update(node="hopper-12"),
                       lambda c: c["config"]["shard"].update(pbs_jobid="999.hopper-m-02"),
                       lambda c: c["metrics"]["black_frame_stop"]["measurement"].update(shape=[224, 224, 3]),
                       lambda c: c["_meta"].update(command=["python3", "-m", "src.envs.pair_shard_approved"]),
                       lambda c: c["_meta"].update(exit_code=1)):
            changed = copy.deepcopy(child)
            change(changed)
            with self.assertRaises(ValueError):
                watch.validate_black_retry(data, JOB, fields, outer, gate, name, "t01v2", changed, record)

    def test_portable_v2_png_is_actually_zero_not_only_a_hash_claim(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            directory = root / "logs/runs/v2/black_frame"
            directory.mkdir(parents=True)
            image, raw = directory / "black.png", directory / "black.json"
            Image.new("RGB", (255, 255)).save(image)

            def proof():
                measured = {"image": str(root / "absent_scratch/black.png"), "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                            "stage": "source", "scene": "FloorPlan1", "measurement": {
                                "shape": [255, 255, 3], "dtype": "uint8", "min": 0, "max": 0, "nonzero_values": 0}}
                raw.write_text(json.dumps(measured))
                raw_sha = hashlib.sha256(raw.read_bytes()).hexdigest()
                return {**measured, "file": str(root / "absent_scratch/black.json"), "sha256": raw_sha,
                        "portable_file": str(raw), "portable_file_sha256": raw_sha,
                        "portable_image": str(image), "portable_image_sha256": measured["image_sha256"]}

            child = {"config": {"shard": {"gate_source": {"sha256": "gate"}}},
                     "metrics": {"kind": "pair_v2_shard", "black_frame_stop": proof()}}
            outer = {"metrics": {"scene_runs": [{"exit_code": 86, "result_path": "results/child.json", "source": {"sha256": "child"}}]}}
            with patch.object(watch, "load_logged", return_value=(child, {"sha256": "child"})):
                watch.child_black_evidence(root, {"environment": {"CVPR_SCRATCH": str(root / "absent_scratch")}}, outer, {"sha256": "gate"}, None)
                # Rehashing a changed nonblack PNG cannot turn it into black proof.
                Image.new("RGB", (255, 255), (1, 0, 0)).save(image)
                child["metrics"]["black_frame_stop"] = proof()
                with self.assertRaises(ValueError):
                    watch.child_black_evidence(root, {"environment": {"CVPR_SCRATCH": str(root / "absent_scratch")}}, outer, {"sha256": "gate"}, None)


if __name__ == "__main__":
    unittest.main()

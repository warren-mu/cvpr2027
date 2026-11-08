"""Synthetic CPU fixtures; commit and invoke through run_logged before use."""
import copy
import hashlib
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

from src.tools.phase_watch import child_black_evidence, entrypoint, require_unused_run, submissions, validate_black_retry

ROOT = Path(__file__).resolve().parents[1]
JOB = "123.hopper-m-02"
UUID = "GPU-11111111-1111-1111-1111-111111111111"


def fixture(kind="scale", gate_black=True):
    name = "pilot_pairs_fixture" if kind == "t01" else "t00b_scale_fixture"
    script = "experiments/t01_parallel_pairs.sh" if kind == "t01" else "experiments/t00b_scale_smoke.sh"
    command = ["bash", script, "--name", name]
    payload = {"name": name, "command": command, "requested_gpu_count": 1}
    fields = {"project": "CFP04-SF-125", "job_state": "F", "Exit_status": "86", "exec_host": "hopper-29/0*12"}
    outer = {"_meta": {"name": name, "exit_code": 86, "command": command}, "config": {"pbs_jobid": JOB}, "metrics": {"cleanup_errors": []}}
    gate_name = name + ("_j123_hopper_m_02_gate" if kind == "t01" else "_gate")
    device = {"logical_index": 0, "gpu_uuid": UUID}
    code = 86 if gate_black else 0
    gate = {"_meta": {"name": gate_name, "exit_code": code,
                      "command": ["python3", "-m", "src.tools.render_gate", "--name", gate_name]},
            "config": {"pbs_jobid": JOB, "node": "hopper-29", "allocation": {
                "pbs_jobid": JOB, "devices": [device], "cuda_visible_devices": UUID}},
            "metrics": {"exit_code": code, "black_frame_detected": gate_black, "passed": not gate_black,
                        "errors": [], "cleanup_errors": [], "completed_gpu_count": 1, "expected_gpu_count": 1,
                        "devices": [{"observation": {"gpu_identity": device,
                            "captures": [{"valid_shape": True, "black": gate_black, "action_success": True,
                                          "valid": not gate_black, "nonzero_rgb_values": 0 if gate_black else 1}
                                         for _ in range(2)]}}]}}
    return payload, fields, outer, gate, gate_name


class WatchFixtures(unittest.TestCase):
    def test_explicit_supported_argv_and_exact_name_only(self):
        name = "t00b_scale_fixture"
        command = ["python3", "src/tools/record_command.py", "--name", name, "--",
                   "bash", "experiments/t00b_scale_smoke.sh", "--name", name, "--gpus", "4"]
        self.assertEqual(entrypoint(command, name, ROOT), "scale")
        self.assertEqual(entrypoint(["bash", "experiments/t00b_gated_diagnostics.sh", "--name", name], name, ROOT),
                         "diagnostics")
        for bad in (["bash", "-c", "echo hello", "--name", name],
                    ["bash", "experiments/t00b_vagen_smoke.sh", "--name", name],
                    command + ["--unrelated-value", name], command[:3] + ["different"] + command[4:]):
            with self.subTest(command=bad), self.assertRaises(ValueError):
                entrypoint(bad, name, ROOT)

    def test_only_complete_same_job_black_gate_is_retryable(self):
        payload, fields, outer, gate, gate_name = fixture()
        self.assertEqual(validate_black_retry(payload, JOB, fields, outer, gate, gate_name, "scale"),
                         "render_gate_black_capture")
        mutations = [lambda f, o, g: f.update(job_state="Q"), lambda f, o, g: f.update(Exit_status="1"),
                     lambda f, o, g: o["_meta"].update(exit_code=1),
                     lambda f, o, g: o["_meta"].update(name="other_run"),
                     lambda f, o, g: g["config"]["allocation"].update(pbs_jobid="other"),
                     lambda f, o, g: g["metrics"].update(black_frame_detected=False),
                     lambda f, o, g: g["metrics"].update(cleanup_errors=["uncertain"]),
                     lambda f, o, g: g["metrics"].update(devices=[]),
                     lambda f, o, g: [capture.update(nonzero_rgb_values=1) for capture in g["metrics"]["devices"][0]["observation"]["captures"]]]
        for mutate in mutations:
            f, o, g = copy.deepcopy((fields, outer, gate))
            mutate(f, o, g)
            with self.assertRaises(ValueError):
                validate_black_retry(payload, JOB, f, o, g, gate_name, "scale")

    def test_t01_capture_requires_clean_same_uuid_audited_child(self):
        payload, fields, outer, gate, gate_name = fixture("t01", gate_black=False)
        record = {"run_name": "pilot_pairs_fixture_scene", "exit_code": 86, "scene": "FloorPlan3",
                  "gpu_uuid": UUID, "logical_index": 0}
        child = {"_meta": {"name": record["run_name"], "exit_code": 86,
                           "command": ["python3", "-m", "src.envs.pair_shard"]},
                 "config": {"shard": {"pbs_jobid": JOB, "gpu_identity": gate["config"]["allocation"]["devices"][0],
                    "child_cuda_visible_devices": UUID, "parent_cuda_visible_devices": UUID, "scene": "FloorPlan3"}},
                 "metrics": {"kind": "t01_scene_collect", "black_frame_detected": True,
                             "capture_black_frame_detected": True, "cleanup_errors": [],
                             "black_frame_stop": {"scene": "FloorPlan3", "stage": "capture", "measurement": {
                                 "shape": [224, 224, 3], "dtype": "uint8", "nonzero_values": 0, "min": 0, "max": 0}}}}
        self.assertEqual(validate_black_retry(payload, JOB, fields, outer, gate, gate_name, "t01", child, record),
                         "t01_scene_black_capture")
        for key, value in (("cleanup_errors", ["unknown"]), ("capture_black_frame_detected", False)):
            changed = copy.deepcopy(child)
            changed["metrics"][key] = value
            with self.assertRaises(ValueError):
                validate_black_retry(payload, JOB, fields, outer, gate, gate_name, "t01", changed, record)
        bad = {**record, "gpu_uuid": "GPU-outside"}
        with self.assertRaises(ValueError):
            validate_black_retry(payload, JOB, fields, outer, gate, gate_name, "t01", child, bad)

    def test_child_raw_evidence_hash_must_still_match(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image, raw = root / "black.png", root / "black.json"
            image.write_bytes(b"synthetic fixture image bytes, not a measured image")
            evidence = {"image": str(image), "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                        "stage": "capture", "measurement": {"nonzero_values": 0}}
            raw.write_text(json.dumps(evidence))
            child = {"config": {"shard": {"gate_source": {"sha256": "gatehash"}}},
                     "metrics": {"black_frame_stop": {"file": str(raw),
                         "sha256": hashlib.sha256(raw.read_bytes()).hexdigest(), **evidence}}}
            payload = {"environment": {"CVPR_SCRATCH": str(root)}}
            outer = {"metrics": {"scene_runs": [{"exit_code": 86, "result_path": "results/child.json",
                                                   "source": {"sha256": "childhash"}}]}}
            with patch("src.tools.phase_watch.load_logged", return_value=(child, {"sha256": "childhash"})):
                self.assertEqual(child_black_evidence(root, payload, outer, {"sha256": "gatehash"}, None)[0], child)
                image.write_bytes(b"changed fixture bytes")
                with self.assertRaises(ValueError):
                    child_black_evidence(root, payload, outer, {"sha256": "gatehash"}, None)

    def test_portable_black_evidence_survives_missing_scratch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            directory = root / "logs/runs/child/black_frame"
            directory.mkdir(parents=True)
            image, raw = directory / "black.png", directory / "black.json"
            scratch = root / "missing_scratch"
            image.write_bytes(b"portable synthetic fixture bytes")
            image_sha = hashlib.sha256(image.read_bytes()).hexdigest()
            original = {"image": str(scratch / "black.png"), "image_sha256": image_sha,
                        "stage": "capture", "measurement": {"nonzero_values": 0}}
            raw.write_text(json.dumps(original))
            raw_sha = hashlib.sha256(raw.read_bytes()).hexdigest()
            stop = {"file": str(scratch / "black.json"), "sha256": raw_sha, **original,
                    "portable_file": str(raw), "portable_file_sha256": raw_sha,
                    "portable_image": str(image), "portable_image_sha256": image_sha}
            child = {"config": {"shard": {"gate_source": {"sha256": "gatehash"}}},
                     "metrics": {"black_frame_stop": stop}}
            payload = {"environment": {"CVPR_SCRATCH": str(scratch)}}
            outer = {"metrics": {"scene_runs": [{"exit_code": 86, "result_path": "results/child.json",
                                                   "source": {"sha256": "childhash"}}]}}
            with patch("src.tools.phase_watch.load_logged", return_value=(child, {"sha256": "childhash"})):
                proof = child_black_evidence(root, payload, outer, {"sha256": "gatehash"}, None)[2]
                self.assertTrue(proof["black_capture"]["portable_copy"])
                self.assertEqual(proof["black_capture"]["image_sha256"], image_sha)
                stop["portable_image_sha256"] = "0" * 64
                with self.assertRaises(ValueError):
                    child_black_evidence(root, payload, outer, {"sha256": "gatehash"}, None)
                stop["portable_image_sha256"] = image_sha
                stop["portable_file"] = str(root / "outside_logs.json")
                with self.assertRaises(ValueError):
                    child_black_evidence(root, payload, outer, {"sha256": "gatehash"}, None)

    def test_uncertain_retry_payload_is_retained_without_job_id(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            directory = root / "logs/pbs/retry"
            directory.mkdir(parents=True)
            (directory / "payload.json").write_text(json.dumps({"budget_batch": "fixture", "retry_of": JOB}))
            (directory / "submission.json").write_text(json.dumps({"stdout": "", "returncode": 1}))
            rows = submissions(root, "fixture", re.compile(r"\b\d+\.hopper-m-\d+\b"))
            self.assertIsNone(rows[0]["job_id"])
            self.assertEqual(rows[0]["payload"]["retry_of"], JOB)
            (root / "results").mkdir()
            (root / "results/existing.json").write_text("{}")
            with self.assertRaises(FileExistsError):
                require_unused_run(root, "existing")


if __name__ == "__main__":
    unittest.main()

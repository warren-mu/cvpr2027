"""Synthetic CPU fixtures; commit and invoke through run_logged before use."""
import copy
import hashlib
import json
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.tools.phase_watch import (child_black_evidence, combine_observations, entrypoint,
                                  final_observations, query_unaccounted, require_unused_run,
                                  submissions, validate_black_retry)

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
    def test_live_query_only_contains_jobs_without_verified_final_rows(self):
        cached = {JOB: {"job_id": JOB, "fields": {"job_state": "F"},
                        "observation_source": "final_accounting"}}
        live = "456.hopper-m-02"
        response = SimpleNamespace(returncode=0, stdout=f"Job Id: {live}\n    job_state = Q\n", stderr="")
        with patch("src.tools.phase_watch.subprocess.run", return_value=response) as run:
            raw = query_unaccounted({JOB, live}, cached)
            self.assertEqual(raw["queried_job_ids"], [live])
            self.assertEqual(run.call_args.args[0][1:], ["-fx", live])
            self.assertEqual(run.call_args.kwargs["timeout"], 180)
            self.assertNotIn(JOB, run.call_args.args[0])
        with patch("src.tools.phase_watch.subprocess.run") as run:
            raw = query_unaccounted({JOB}, cached)
            run.assert_not_called()
            self.assertIsNone(raw["command"])
        bad = copy.deepcopy(cached)
        bad[JOB]["fields"]["job_state"] = "R"
        with self.assertRaises(ValueError):
            query_unaccounted({JOB}, bad)

    def test_cached_fields_require_unique_same_job_terminal_accounting(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            (root / "logs/runs").mkdir(parents=True)
            log = root / "logs/runs/accounting.log"
            fields = {"job_state": "F", "Exit_status": "86", "project": "CFP04-SF-125"}
            row = {"job_id": JOB, "qstat_exit": 0, "fields": fields}
            data = {"config": {"jobs": [row]}, "metrics": {"complete_allocation_accounting": True,
                    "allocation_gpu_hours_by_job": {JOB: 0.5}},
                    "_meta": {"exit_code": 0, "command": ["python3", "src/tools/pbs_usage.py"]}}
            ref = {"path": "results/accounting.json", "log_path": "logs/runs/accounting.log", "sha256": "fixture"}
            final = {JOB: {"gpu_hours": 0.5, "accounting_result": "results/accounting.json"}}
            log.write_text(json.dumps(data["config"], indent=2))
            with patch("src.tools.phase_watch.load_logged", return_value=(data, ref)):
                cached = final_observations(root, final, None)
                self.assertEqual(cached[JOB]["fields"], fields)
                self.assertEqual(cached[JOB]["observation_source"], "final_accounting")
                self.assertEqual(cached[JOB]["source"], ref)
            bad_rows = [[], [row, copy.deepcopy(row)], [{**row, "job_id": "other.hopper-m-02"}],
                        [{**row, "fields": {**fields, "job_state": "R"}}], [{**row, "qstat_exit": 1}]]
            for rows in bad_rows:
                changed = copy.deepcopy(data)
                changed["config"]["jobs"] = rows
                log.write_text(json.dumps(changed["config"], indent=2))
                with patch("src.tools.phase_watch.load_logged", return_value=(changed, ref)), self.assertRaises(ValueError):
                    final_observations(root, final, None)
            log.write_text("different original fields")
            with patch("src.tools.phase_watch.load_logged", return_value=(data, ref)), self.assertRaises(ValueError):
                final_observations(root, final, None)

    def test_missing_or_ambiguous_live_response_never_becomes_cached_fields(self):
        source = {"path": "logs/runs/fixture/qstat.json", "sha256": "fixture"}
        submit = SimpleNamespace(pbs_blocks=lambda text, heading: {JOB: {"job_state": "Q"}})
        raw = {"queried_job_ids": [JOB], "stdout": f"Job Id: {JOB}\n    job_state = Q\n"}
        rows = combine_observations({JOB}, {}, raw, source, submit)
        self.assertEqual(rows[JOB]["observation_source"], "live_qstat")
        for text in ("", raw["stdout"] * 2, "Job Id: 999.hopper-m-02\n"):
            with self.assertRaises(ValueError):
                combine_observations({JOB}, {}, {**raw, "stdout": text}, source, submit)

    def test_explicit_supported_argv_and_exact_name_only(self):
        name = "t00b_scale_fixture"
        command = ["python3", "src/tools/record_command.py", "--name", name, "--",
                   "bash", "experiments/t00b_scale_smoke.sh", "--name", name, "--gpus", "4"]
        self.assertEqual(entrypoint(command, name, ROOT), "scale")
        self.assertEqual(entrypoint(["bash", "experiments/t00b_gated_diagnostics.sh", "--name", name], name, ROOT),
                         "diagnostics")
        self.assertEqual(entrypoint(["bash", "experiments/t01_collision_pack.sh", "--name", name], name, ROOT),
                         "probe_pack")
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
        approved = copy.deepcopy(child)
        approved["_meta"]["command"][2] = "src.envs.pair_shard_approved"
        self.assertEqual(validate_black_retry(payload, JOB, fields, outer, gate, gate_name, "t01", approved, record),
                         "t01_scene_black_capture")
        # Actual approved shard shape uses sys.executable and unbuffered output.
        approved["_meta"]["command"] = ["/scratch/fixture/env/bin/python3", "-u", "-m",
                                       "src.envs.pair_shard_approved", "--name", record["run_name"]]
        self.assertEqual(validate_black_retry(payload, JOB, fields, outer, gate, gate_name, "t01", approved, record),
                         "t01_scene_black_capture")
        legacy_unbuffered = copy.deepcopy(approved)
        legacy_unbuffered["_meta"]["command"][3] = "src.envs.pair_shard"
        self.assertEqual(validate_black_retry(payload, JOB, fields, outer, gate, gate_name, "t01", legacy_unbuffered, record),
                         "t01_scene_black_capture")
        forbidden = [
            ["python3", "-u", "-m", "src.envs.pair_shard_unreviewed"],
            ["python3", "-u", "-m", "unrelated.module", "--value", "src.envs.pair_shard_approved"],
            ["python3", "other.py", "-m", "src.envs.pair_shard_approved"],
            ["python3", "-c", "print('fixture')", "-m", "src.envs.pair_shard"],
            ["bash", "-u", "-m", "src.envs.pair_shard_approved"],
            ["not-python3", "-m", "src.envs.pair_shard"],
            ["python3", "-u", "-u", "-m", "src.envs.pair_shard"],
            ["python3", "-m"], None,
        ]
        for command in forbidden:
            unknown = copy.deepcopy(child)
            unknown["_meta"]["command"] = command
            with self.subTest(command=command), self.assertRaises(ValueError):
                validate_black_retry(payload, JOB, fields, outer, gate, gate_name, "t01", unknown, record)
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

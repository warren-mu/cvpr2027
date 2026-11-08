"""Synthetic CPU fixtures only; execute after commit through run_logged.

No real run JSON, scheduler, Ray, simulator, training or GPU is accessed.
"""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from src.tools import t00b_training_summary as summary


class TrainingSummaryTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()
        self.output = self.root / "results/summary.json"
        for owner in (summary, summary.provenance):
            handle = patch.object(owner, "ROOT", self.root)
            handle.start()
            self.addCleanup(handle.stop)

    def reference(self, name):
        return {"file": "results/" + name + ".json",
                "sha256": hashlib.sha256(name.encode()).hexdigest(),
                "run_name": name, "exit_code": 0}

    def training(self, gpus, variant="baseline"):
        name = f"{variant}_{gpus}"
        identity = f"{1000 + gpus}.hopper-m-02"
        reference = self.reference(name)
        uuids = [f"GPU-{gpus}-{index}" for index in range(gpus)]
        config = {
            "name": name, "pbs_jobid": identity,
            "variant": variant, "baseline": "official_vagen_base_navigation",
            "allocation_gpu_count": gpus, "rollout_tensor_parallel_size": 1 if gpus == 2 else 4,
            "additional_critic_offload": False,
            "vagen_commit": "a" * 40, "verl_commit": "b" * 40,
            "training_config_changes": {"trainer.total_training_steps": 21 if variant == "baseline" else 6},
            "allocation": {"pbs_jobid": identity, "devices": [
                {"logical_index": index, "gpu_uuid": uuid} for index, uuid in enumerate(uuids)]},
            "same_job_baseline": None,
        }
        if variant == "vllm04":
            baseline = self.reference(f"baseline_{gpus}")
            config["same_job_baseline"] = {"path": str(self.root / baseline["file"]), "sha256": baseline["sha256"]}
        count = 20 if variant == "baseline" else 5
        metrics = {
            "completed_training_steps": count, "wrapper_exit_code": 0, "training_command_exit_code": 0,
            "median_step_seconds": {2: 8.0, 4: 4.0, 8: 2.0}[gpus],
            "median_rollout_seconds": 4.0 if variant == "baseline" else 2.0,
            "gpu_resources_by_uuid": {uuid: {"memory_peak_mib_sampled": 100, "valid_memory_samples": 1} for uuid in uuids},
            "process_tree_rss_peak_bytes": 123456,
            "actual_new_rgb_frames_per_training_step": {"1": 9},
            "render_frame_accounting": {"valid": True, "initialization": {"fresh_main_rgb": 3}},
            # These intentionally arbitrary fixture values must be COPIED, not recomputed.
            "full_training_estimates": {
                "100": {"configured_steps": 100, "estimated_wall_hours": 12.5, "estimated_gpu_hours": 25.0},
                "300": {"configured_steps": 300, "estimated_wall_hours": 37.5, "estimated_gpu_hours": 75.0}},
        }
        data = {"metrics": metrics, "config": config, "acceptance_passed": True,
                "_meta": {"name": name, "exit_code": 0, "start": "synthetic", "git_commit": "c" * 40},
                "completed_step_timings": {"1": {"timing_s": {"step": 8.0, "gen": 4.0}}},
                "measurement_notes": ["synthetic fixture, not research measurements"]}
        return data, reference

    def accounting(self, name="account", pbs_exit="0"):
        jobs, hours = [], {}
        for gpus in (2, 4, 8):
            identity = f"{1000 + gpus}.hopper-m-02"
            fields = {"job_state": "F", "Resource_List.ngpus": str(gpus),
                      "Resource_List.walltime": "02:00:00", "resources_used.walltime": "01:00:00",
                      "run_count": "1", "exec_host": "fixture-node/0", "Exit_status": pbs_exit,
                      "project": "CFP04-SF-125", "queue": "smallx" if gpus == 2 else "mediumx" if gpus == 4 else "largex",
                      "Resource_List.queue_charge_rate": "1", "Resource_List.mem": "unknown-unit",
                      "resources_used.mem": "123kb", "exec_vnode": "(fixture-node)",
                      "qtime": "Sun Nov  8 10:00:00 2026", "etime": "Sun Nov  8 10:00:00 2026",
                      "stime": "Sun Nov  8 10:01:00 2026"}
            jobs.append({"job_id": identity, "qstat_exit": 0, "fields": fields})
            hours[identity] = float(gpus)
        return {"metrics": {"allocation_gpu_hours_by_job": hours}, "config": {"jobs": jobs}}, self.reference(name)

    def inputs(self):
        declarations = summary.parse_slots(
            [f"{n}=results/baseline_{n}.json" for n in (2, 4, 8)],
            [f"{n}=results/vllm04_{n}.json" for n in (2, 4, 8)])
        loaded = {}
        for variant in ("baseline", "vllm04"):
            for n in (2, 4, 8):
                data, reference = self.training(n, variant)
                loaded[reference["file"]] = (data, reference)
        loaded["results/account.json"] = self.accounting()
        return declarations, loaded

    def summarize(self, declarations, loaded, history=None, accounts=None):
        return summary.summarize(declarations, history or [], accounts if accounts is not None else ["results/account.json"], loaded, self.output)

    def test_six_explicit_slots_copy_evidence_and_estimates_without_recomputing(self):
        declarations, loaded = self.inputs()
        result = self.summarize(declarations, loaded)
        self.assertTrue(result["aggregation_program_completed"])
        self.assertTrue(result["all_six_selected_runs_eligible"])
        self.assertIsNone(result["final_task_acceptance"])
        self.assertEqual(result["comparisons"]["scaling"][0]["speedup"], 2.0)
        self.assertEqual(result["comparisons"]["scaling"][0]["efficiency"], 1.0)
        self.assertEqual(result["comparisons"]["variant_generation"][0]["baseline_gen_over_vllm04_gen"], 2.0)
        self.assertEqual(result["comparisons"]["variant_generation"][0]["vllm04_minus_baseline_gen_seconds"], -2.0)
        row = result["slots"]["baseline_2"]
        self.assertEqual(row["raw_source_record"], loaded["results/baseline_2.json"][0])
        self.assertEqual(row["eligible_estimates"], loaded["results/baseline_2.json"][0]["metrics"]["full_training_estimates"])
        self.assertEqual(result["accounting"]["unique_accounted_job_count"], 3)
        self.assertEqual(result["accounting"]["total_allocation_gpu_hours"], 14.0)
        self.assertEqual(row["final_accounting"]["queue_wait_seconds"], 60)
        self.assertIsNone(row["final_accounting"]["memory_within_requested"])
        self.assertEqual(row["final_accounting"]["raw_scheduler_fields"]["Resource_List.mem"], "unknown-unit")

    def test_old_two_gpu_schema_does_not_require_optional_scale_count(self):
        data, _ = self.training(2)
        del data["config"]["allocation_gpu_count"]
        self.assertEqual(summary.eligibility(data, 2, "baseline"), [])

    def test_missing_slot_stays_null_even_with_successful_history(self):
        declarations, loaded = self.inputs()
        loaded["results/old_success.json"] = loaded["results/baseline_8.json"]
        loaded["results/baseline_8.json"] = (None, None)
        result = self.summarize(declarations, loaded, ["results/old_success.json"])
        row = result["slots"]["baseline_8"]
        self.assertIsNone(row["raw_source_record"])
        self.assertFalse(row["eligible"])
        self.assertIsNone(row["eligible_estimates"])
        self.assertIsNone(result["comparisons"]["scaling"][1]["speedup"])
        self.assertEqual(result["research_status"], "incomplete")
        self.assertEqual(len(result["history"]), 1)

    def test_failed_run_with_twenty_updates_retains_raw_but_no_comparison(self):
        declarations, loaded = self.inputs()
        data = loaded["results/baseline_4.json"][0]
        data["metrics"]["training_command_exit_code"] = 1
        result = self.summarize(declarations, loaded)
        row = result["slots"]["baseline_4"]
        self.assertFalse(row["eligible"])
        self.assertEqual(row["raw_source_record"]["metrics"]["completed_training_steps"], 20)
        self.assertIsNotNone(row["raw_source_record"]["metrics"]["full_training_estimates"])
        self.assertIsNone(row["eligible_estimates"])
        self.assertIsNone(result["comparisons"]["scaling"][0]["speedup"])

    def test_each_exit_and_acceptance_is_required_not_truthy(self):
        for owner, key, value in (("_meta", "exit_code", 1), ("metrics", "wrapper_exit_code", 1),
                                  ("metrics", "training_command_exit_code", True)):
            with self.subTest(owner=owner, key=key):
                data, _ = self.training(2)
                data[owner][key] = value
                self.assertTrue(summary.eligibility(data, 2, "baseline"))
        data, _ = self.training(2)
        data["acceptance_passed"] = 1
        self.assertIn("original_acceptance_not_true", summary.eligibility(data, 2, "baseline"))

    def test_completed_update_thresholds_and_nonfinite_timing(self):
        for variant, threshold in (("baseline", 20), ("vllm04", 5)):
            for count, valid in ((threshold - 1, False), (threshold, True)):
                data, _ = self.training(2, variant)
                data["metrics"]["completed_training_steps"] = count
                self.assertEqual(not summary.eligibility(data, 2, variant), valid)
        for value in (0, -1, float("nan"), float("inf"), True, None):
            with self.subTest(value=value):
                data, _ = self.training(2)
                data["metrics"]["median_step_seconds"] = value
                self.assertIn("median_step_not_finite_positive", summary.eligibility(data, 2, "baseline"))

    def test_declared_variant_topology_and_uuid_evidence_must_match(self):
        for mutate in (
            lambda d: d["config"].update(variant="vllm04"),
            lambda d: d["config"].update(rollout_tensor_parallel_size=1),
            lambda d: d["config"].update(additional_critic_offload=True),
            lambda d: d["config"].update(allocation_gpu_count=8),
            lambda d: d["metrics"]["gpu_resources_by_uuid"].pop("GPU-4-0"),
            lambda d: d["config"]["allocation"]["devices"][0].update(gpu_uuid="wrong"),
            lambda d: d["config"].update(pbs_jobid="1004"),
        ):
            data, _ = self.training(4)
            mutate(data)
            self.assertTrue(summary.eligibility(data, 4, "baseline"))

    def test_variant_must_reference_explicit_baseline_file_sha_and_job(self):
        for field, value in (("path", "results/other.json"), ("sha256", "0" * 64), ("pbs_jobid", "9999.hopper-m-02")):
            with self.subTest(field=field):
                declarations, loaded = self.inputs()
                config = loaded["results/vllm04_2.json"][0]["config"]
                if field == "pbs_jobid":
                    config[field] = value
                    config["allocation"][field] = value
                else:
                    config["same_job_baseline"][field] = value
                result = self.summarize(declarations, loaded)
                self.assertFalse(result["slots"]["vllm04_2"]["eligible"])
                self.assertIsNone(result["comparisons"]["variant_generation"][0]["baseline_gen_over_vllm04_gen"])

    def test_baseline_stays_eligible_when_later_variant_and_whole_pbs_fail(self):
        declarations, loaded = self.inputs()
        loaded["results/account.json"] = self.accounting(pbs_exit="1")
        variant = loaded["results/vllm04_2.json"][0]
        variant["_meta"]["exit_code"] = 1
        variant["metrics"]["wrapper_exit_code"] = 1
        variant["metrics"]["training_command_exit_code"] = 1
        variant["acceptance_passed"] = False
        result = self.summarize(declarations, loaded)
        self.assertTrue(result["slots"]["baseline_2"]["eligible"])
        self.assertEqual(result["comparisons"]["scaling"][0]["speedup"], 2.0)
        self.assertFalse(result["slots"]["vllm04_2"]["eligible"])
        self.assertFalse(result["all_six_selected_runs_eligible"])
        self.assertEqual(result["slots"]["baseline_2"]["final_accounting"]["pbs_exit_status"], "1")
        self.assertEqual(result["accounting"]["unique_accounted_job_count"], 3)
        self.assertEqual(result["accounting"]["total_allocation_gpu_hours"], 14.0)

    def test_accounting_deduplicates_and_rejects_conflicting_final_charge_or_wait(self):
        first = self.accounting()
        second = self.accounting("again")
        result = summary.link_accounting([first, second], [], [])
        self.assertEqual(result["unique_accounted_job_count"], 3)
        self.assertEqual(result["total_allocation_gpu_hours"], 14.0)
        for field, value in (("Resource_List.queue_charge_rate", "2"), ("resources_used.mem", "124kb"),
                             ("stime", "Sun Nov  8 10:02:00 2026")):
            changed = copy.deepcopy(second)
            changed[0]["config"]["jobs"][0]["fields"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                summary.link_accounting([first, changed], [], [])

    def test_missing_accounting_never_becomes_zero_or_complete(self):
        declarations, loaded = self.inputs()
        result = self.summarize(declarations, loaded, accounts=[])
        self.assertIsNone(result["accounting"]["total_allocation_gpu_hours"])
        self.assertIsNone(result["accounting"]["known_allocation_gpu_hours_subtotal"])
        self.assertFalse(result["accounting"]["complete_for_supplied_run_jobs"])
        loaded["results/missing_account.json"] = (None, None)
        result = self.summarize(declarations, loaded, accounts=["results/account.json", "results/missing_account.json"])
        self.assertIsNone(result["accounting"]["total_allocation_gpu_hours"])
        self.assertEqual(result["accounting"]["known_allocation_gpu_hours_subtotal"], 14.0)

    def test_selected_gpu_count_must_match_final_job_even_when_bill_is_self_consistent(self):
        declarations, loaded = self.inputs()
        account = loaded["results/account.json"][0]
        # A self-consistent smaller bill must not be attached to a four-GPU run.
        job = account["config"]["jobs"][1]
        job["fields"]["Resource_List.ngpus"] = "2"
        account["metrics"]["allocation_gpu_hours_by_job"][job["job_id"]] = 2.0
        with self.assertRaisesRegex(ValueError, "Selected training slot disagrees"):
            self.summarize(declarations, loaded)

    def write_source(self, relative="results/fixture.json", data=None):
        data = copy.deepcopy(data or self.training(2)[0])
        data["_meta"]["metrics_sha256"] = summary.provenance.metrics_hash(data["metrics"])
        data["_meta"]["log_path"] = "logs/runs/fixture.log"
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
        log = self.root / data["_meta"]["log_path"]
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("FINAL_METRICS_JSON: " + json.dumps(data["metrics"]) + "\n")
        auditor = types.SimpleNamespace(audit_file=lambda _: {"fail": [], "warn": ["synthetic fixture"]})
        return path, log, auditor

    def test_source_audit_hash_final_and_missing_semantics(self):
        path, log, auditor = self.write_source()
        loaded, sources, audits, missing = summary.load_inputs([str(path), "results/absent.json"], self.output, auditor)
        self.assertEqual(sources[0]["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(audits[0]["warn"], ["synthetic fixture"])
        self.assertEqual(loaded["results/absent.json"], (None, None))
        self.assertEqual(missing[0]["requested_source"], "results/absent.json")
        log.write_text('FINAL_METRICS_JSON: {}\n')
        with self.assertRaisesRegex(ValueError, "final metrics"):
            summary.load_inputs([str(path)], self.output, auditor)
        _, _, auditor = self.write_source()
        data = json.loads(path.read_text())
        data["metrics"]["completed_training_steps"] = 19
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            summary.load_inputs([str(path)], self.output, auditor)

    def test_duplicate_paths_run_identities_and_auditor_failure_are_rejected(self):
        path, _, auditor = self.write_source()
        with self.assertRaisesRegex(ValueError, "Duplicate explicit"):
            summary.load_inputs([str(path), "results/fixture.json"], self.output, auditor)
        other = self.root / "results/copy.json"
        other.write_bytes(path.read_bytes())
        verified = []
        with self.assertRaisesRegex(ValueError, "Duplicate source run"):
            summary.load_inputs([str(path), str(other)], self.output, auditor, sources=verified)
        self.assertEqual(len(verified), 1)  # Prior valid evidence remains available on later failure.
        auditor.audit_file = lambda _: {"fail": ["synthetic provenance corruption"], "warn": []}
        with self.assertRaisesRegex(ValueError, "provenance corruption"):
            summary.load_inputs([str(path)], self.output, auditor)

    def test_slot_and_path_ambiguity_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate slot"):
            summary.parse_slots(["2=results/a.json", "2=results/b.json"], [])
        with self.assertRaisesRegex(ValueError, "all six"):
            summary.parse_slots(["2=results/a.json"], [])
        for path in ("../outside.json", "logs/runs/not_results.json", "results/summary.json"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                summary.source_path(path, self.output)


if __name__ == "__main__":
    unittest.main()

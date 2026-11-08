"""Pure synthetic accounting cases; no scheduler, remote query or real totals."""
from contextlib import redirect_stdout
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.tools import pbs_wait_table as report

JOB = "123.hopper-m-02"


def fields():
    return {"job_state": "F", "project": "CFP04-SF-125", "Job_Name": "cvpr27_fixture",
        "Job_Owner": "fixture@hopper-l-01", "run_count": "1", "Exit_status": "1", "queue": "smallx",
        "Resource_List.ngpus": "2", "Resource_List.select": "1:ngpus=2:ncpus=24:mem=450GB",
        "Resource_List.walltime": "01:00:00", "resources_used.walltime": "00:30:00",
        "Resource_List.queue_charge_rate": "2", "exec_host": "hopper-11/0*24", "exec_vnode": "(hopper-11:ncpus=24)",
        "qtime": "Sun Nov  8 12:00:00 2026", "etime": "Sun Nov  8 12:00:30 2026",
        "stime": "Sun Nov  8 12:01:30 2026"}


def source(root, name="account_fixture", *, raw=None, hours=1.0):
    raw = copy.deepcopy(fields() if raw is None else raw)
    wait = report.queue_wait_record(raw)
    data = {"_meta": {"name": name, "exit_code": 0,
        "command": ["/scratch/env/bin/python3", "src/tools/pbs_usage.py", "--name", name, JOB],
        "log_path": f"logs/runs/{name}.log"},
        "config": {"jobs": [{"job_id": JOB, "qstat_exit": 0, "fields": raw, **wait}],
                   "errors": [], "cancelled_evidence": None},
        "metrics": {"allocation_gpu_hours_by_job": {JOB: hours}, "complete_allocation_accounting": True,
                    "total_allocation_gpu_hours": hours,
                    "queue_wait_seconds_by_job": {JOB: wait["queue_wait_seconds"]},
                    "eligible_wait_seconds_by_job": {JOB: wait["eligible_wait_seconds"]},
                    "verified_never_started_cancelled_jobs": []}}
    reference = {"file": f"results/{name}.json", "sha256": "a" * 64,
                 "metrics_sha256": hashlib.sha256(json.dumps(data["metrics"], sort_keys=True).encode()).hexdigest()}
    save_log(root, data)
    return data, reference


def save_log(root, data):
    path = root / data["_meta"]["log_path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data["config"], indent=2) + "\nFINAL_METRICS_JSON: " + json.dumps(data["metrics"]) + "\n")


class QueueReportTests(unittest.TestCase):
    def test_exact_producer_and_argument_identity(self):
        root = Path("/fixture/repo")
        valid = ["python3", "-u", "src/tools/pbs_usage.py", "--name", "account", JOB]
        self.assertEqual(report.producer_arguments(valid, root), ("account", [JOB], None))
        for argv in (["bash", *valid[1:]], ["python3", "fake.py", *valid[3:]],
                     ["python3", "-c", "src/tools/pbs_usage.py", JOB], valid + [JOB], valid + ["--name", "again"]):
            with self.assertRaises(ValueError):
                report.producer_arguments(argv, root)

    def test_actual_wait_units_rate_and_failed_experiment_cost_are_retained(self):
        row = report.job_record({"job_id": JOB, "qstat_exit": 0, "fields": fields()}, 1.0)
        self.assertEqual(row["queue_wait_seconds"], 90)
        self.assertEqual(row["queue_wait_minutes"], 1.5)
        self.assertEqual(row["eligible_wait_seconds"], 60)
        self.assertEqual(row["pbs_timestamps_iso8601"]["stime"], "2026-11-08T12:01:30+08:00")
        self.assertEqual(row["allocation_gpu_hours"], 1.0)
        self.assertEqual(row["scheduler_charged_gpu_hours"], 2.0)
        self.assertEqual(row["pbs_exit_status"], "1")  # An unsuccessful experiment still consumed its allocation.
        self.assertEqual(row["node"], "hopper-11")

    def test_missing_or_multiple_run_wait_is_unknown_not_zero(self):
        for change in ({"run_count": "2"}, {"run_count": None}, {"stime": None},
                       {"qtime": "Sun Nov  8 13:00:00 2026"}):
            raw = {**fields(), **change}
            row = report.job_record({"job_id": JOB, "qstat_exit": 0, "fields": raw}, 1.0)
            self.assertIsNone(row["queue_wait_seconds"])
            self.assertIsNone(row["queue_wait_minutes"])
            self.assertIsNotNone(row["queue_wait_unavailable_reason"])

    def cancelled(self):
        raw = fields()
        for key in ("run_count", "stime", "exec_host", "exec_vnode", "resources_used.walltime"):
            raw.pop(key)
        return raw

    def test_never_started_requires_explicit_zero_run_or_authenticated_chain(self):
        raw = self.cancelled()
        job = {"job_id": JOB, "qstat_exit": 0, "fields": raw}
        with self.assertRaisesRegex(ValueError, "not proof of zero"):
            report.job_record(job, 0.0)
        for proof in ({"postflight_fields": copy.deepcopy(raw)}, None):
            if proof is None:
                raw["run_count"] = "0"
            row = report.job_record(job, 0.0, proof)
            self.assertTrue(row["never_started"])
            self.assertIsNone(row["node"])
            self.assertIsNone(row["queue_wait_seconds"])
            self.assertEqual(row["allocation_gpu_hours"], 0.0)
        raw["stime"] = "Sun Nov  8 12:00:00 2026"
        with self.assertRaises(ValueError):
            report.job_record(job, 0.0, {"postflight_fields": self.cancelled()})

    def test_recorded_cancellation_rechecks_chain_and_source_references(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = self.cancelled()
            data, reference = source(root, raw=raw, hours=0.0)
            paths = ["results/before.json", "results/qdel.json", "results/after.json"]
            data["_meta"]["command"] += ["--cancelled-evidence", *paths]
            data["config"]["cancelled_evidence"] = paths
            data["metrics"]["verified_never_started_cancelled_jobs"] = [JOB]
            refs = [{"file": path, "sha256": "c" * 64} for path in paths]
            data["sources"] = refs
            save_log(root, data)
            with patch.object(report, "verify_evidence", return_value=({JOB: {"postflight_fields": raw}}, refs)) as verify:
                metrics, _ = report.aggregate([(data, reference)], root)
                verify.assert_called_once_with(root, paths)
                self.assertEqual(metrics["total_allocation_gpu_hours"], 0.0)
                self.assertIsNone(metrics["jobs"][0]["run_count_raw"])
                self.assertIsNone(metrics["jobs"][0]["queue_wait_seconds"])
            data["sources"] = []
            with patch.object(report, "verify_evidence", return_value=({JOB: {"postflight_fields": raw}}, refs)), self.assertRaises(ValueError):
                report.aggregate([(data, reference)], root)

    def test_matching_final_histories_count_once_and_conflicts_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first, second = source(root, "first"), source(root, "second")
            metrics, _ = report.aggregate([first, second], root)
            self.assertEqual(metrics["unique_job_count"], 1)
            self.assertEqual(metrics["total_allocation_gpu_hours"], 1.0)
            self.assertEqual(metrics["total_scheduler_charged_gpu_hours"], 2.0)
            self.assertEqual(metrics["jobs"][0]["source_files"], ["results/first.json", "results/second.json"])
            changes = [({"qtime": "Sun Nov  8 11:59:00 2026"}, 1.0),
                       ({"Resource_List.ngpus": "4"}, 2.0),
                       ({"Resource_List.queue_charge_rate": "3"}, 1.0),
                       ({"resources_used.walltime": "00:15:00"}, .5)]
            for change, hours in changes:
                conflicting = source(root, "conflicting", raw={**fields(), **change}, hours=hours)
                with self.assertRaisesRegex(ValueError, "Conflicting final"):
                    report.aggregate([first, conflicting], root)
            with self.assertRaisesRegex(ValueError, "Duplicate explicit"):
                report.aggregate([first, first], root)

    def test_missing_recorded_rate_never_uses_another_field_or_assumes_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = fields()
            del raw["Resource_List.queue_charge_rate"]
            raw["am_job_cache"] = "queue_charge_rate=1"
            data, ref = source(Path(tmp), raw=raw)
            metrics, _ = report.aggregate([(data, ref)], Path(tmp))
            self.assertFalse(metrics["complete_recorded_rate_charges"])
            self.assertIsNone(metrics["total_scheduler_charged_gpu_hours"])
            self.assertIsNone(metrics["jobs"][0]["queue_charge_rate"])
            self.assertIsNone(metrics["jobs"][0]["scheduler_charged_gpu_hours"])

    def test_explicit_zero_rate_is_known_while_invalid_rates_fail(self):
        raw = {**fields(), "Resource_List.queue_charge_rate": "0"}
        row = report.job_record({"job_id": JOB, "qstat_exit": 0, "fields": raw}, 1.0)
        self.assertEqual(row["queue_charge_rate"], 0.0)
        self.assertEqual(row["scheduler_charged_gpu_hours"], 0.0)
        self.assertIsNone(row["charged_hours_unavailable_reason"])
        for invalid in ("-1", "nan", "inf", True):
            raw["Resource_List.queue_charge_rate"] = invalid
            with self.assertRaises(ValueError):
                report.job_record({"job_id": JOB, "qstat_exit": 0, "fields": raw}, 1.0)

    def test_all_unique_jobs_survive_a_missing_rate_or_wait(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = source(root, "first")
            raw = fields()
            raw.pop("Resource_List.queue_charge_rate")
            raw.pop("stime")
            data, reference = source(root, "second", raw=raw)
            another = "124.hopper-m-02"
            data["_meta"]["command"][-1] = another
            data["config"]["jobs"][0]["job_id"] = another
            for key in ("allocation_gpu_hours_by_job", "queue_wait_seconds_by_job", "eligible_wait_seconds_by_job"):
                data["metrics"][key][another] = data["metrics"][key].pop(JOB)
            save_log(root, data)
            metrics, _ = report.aggregate([first, (data, reference)], root)
            self.assertEqual([row["job_id"] for row in metrics["jobs"]], [JOB, another])
            self.assertEqual(metrics["total_allocation_gpu_hours"], 2.0)
            self.assertIsNone(metrics["total_scheduler_charged_gpu_hours"])
            self.assertEqual(metrics["known_scheduler_charged_gpu_hours_subtotal"], 2.0)
            self.assertEqual(metrics["queue_wait_unavailable_job_ids"], [another])
            self.assertEqual(metrics["charge_unavailable_job_ids"], [another])

    def test_config_log_identity_wait_and_cost_corruption_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, reference = source(root)
            altered = copy.deepcopy(data)
            altered["config"]["jobs"][0]["fields"]["qtime"] = "Sun Nov  8 11:00:00 2026"
            with self.assertRaisesRegex(ValueError, "source log"):
                report.aggregate([(altered, reference)], root)
            altered = copy.deepcopy(data)
            altered["metrics"]["queue_wait_seconds_by_job"][JOB] = 0
            with self.assertRaises(ValueError):
                report.aggregate([(altered, reference)], root)
            altered = copy.deepcopy(data)
            altered["_meta"]["name"] = "another"
            with self.assertRaises(ValueError):
                report.aggregate([(altered, reference)], root)
            with self.assertRaises(ValueError):
                report.job_record(data["config"]["jobs"][0], 0.0)

    def test_main_writes_identical_final_and_p8_compatible_sources_without_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, reference = source(root)
            argv = ["pbs_wait_table", "--name", "queue_fixture", "--source", reference["file"]]
            capture = io.StringIO()
            with patch.object(report, "ROOT", root), patch.object(report.sys, "argv", argv), \
                 patch.object(report, "audited_result", return_value=(data, reference)), redirect_stdout(capture):
                code = report.main()
            self.assertEqual(code, 0)
            result = json.loads((root / "results/queue_fixture.json").read_text())
            final = capture.getvalue().strip().split("FINAL_METRICS_JSON: ")[-1]
            self.assertEqual(json.loads(final), result["metrics"])
            self.assertEqual(result["sources"], [reference])
            self.assertTrue(all("file" in ref and "metrics_sha256" in ref for ref in result["sources"]))
            md = root / result["metrics"]["markdown_report"]["file"]
            self.assertEqual(hashlib.sha256(md.read_bytes()).hexdigest(), result["metrics"]["markdown_report"]["sha256"])
            self.assertIn("../../results/account_fixture.json", md.read_text())
            with patch.object(report, "ROOT", root), patch.object(report.sys, "argv", argv), self.assertRaises(FileExistsError):
                report.main()


if __name__ == "__main__":
    unittest.main()

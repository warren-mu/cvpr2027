"""CPU-only cancellation accounting fixtures; run after commit via run_logged."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.tools.pbs_cancel_evidence import (flatten_fields, match_terminal, not_started,
                                           validate_chain, verify_evidence)

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("cancel_submit", ROOT / "scripts/pbs/submit_phase.py")
submit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(submit)
JOB, OTHER = "123.hopper-m-02", "124.hopper-m-02"
USER = "fixture_owner"


def fixture():
    before = {"job_state": "Q", "Job_Name": "cvpr27_fixture", "Job_Owner": USER + "@hopper-l-01",
              "project": "CFP04-SF-125", "Resource_List": {"ngpus": 2, "select": "1:ngpus=2:ncpus=24:mem=450GB",
              "walltime": "00:30:00", "queue_charge_rate": 1}}
    after = {**copy.deepcopy(before), "job_state": "F", "comment": "terminated"}
    commands = [["qstat", "-xf", "-F", "json", JOB, OTHER], ["qdel", JOB],
                ["qstat", "-xf", "-F", "json", JOB, OTHER]]
    records = []
    logs = [json.dumps({"Jobs": {JOB: before, OTHER: {"job_state": "R"}}}), "",
            json.dumps({"Jobs": {JOB: after, OTHER: {"job_state": "R"}}})]
    for index, command in enumerate(commands):
        metrics = {"command_exit_code": 0, "elapsed_seconds": .5}
        name = "cancel_fixture_" + str(index)
        records.append({"config": {"command": command}, "metrics": metrics, "error": None,
                        "_meta": {"name": name, "host": "hopper-l-01", "start": f"2026-11-09T01:00:0{index * 2}",
                                  "end": f"2026-11-09T01:00:0{index * 2 + 1}", "exit_code": 0,
                                  "command": ["python3", "src/tools/record_command.py", "--name", name, "--", *command],
                                  "metrics_sha256": hashlib.sha256(json.dumps(metrics, sort_keys=True).encode()).hexdigest(),
                                  "log_path": "logs/runs/" + name + ".log"}})
    return records, logs


def materialize(root, records, logs):
    (root / "results").mkdir(parents=True)
    (root / "logs/runs").mkdir(parents=True)
    paths = []
    for record, text in zip(records, logs):
        path = Path("results") / (record["_meta"]["name"] + ".json")
        (root / path).write_text(json.dumps(record))
        (root / record["_meta"]["log_path"]).write_text(text + "\nFINAL_METRICS_JSON: " + json.dumps(record["metrics"]) + "\n")
        paths.append(str(path))
    return paths


class CancellationEvidenceTests(unittest.TestCase):
    def test_exact_module_recorder_is_authenticated_without_rewriting_source(self):
        records, logs = fixture()
        for record in records:
            argv = record['_meta']['command']
            record['_meta']['command'] = [argv[0], '-m', 'src.tools.record_command', *argv[2:]]
        auditor = SimpleNamespace(audit_file=lambda path: {'fail': [], 'warn': []})
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            paths = materialize(root, records, logs)
            verified, _ = verify_evidence(root, paths, auditor, USER)
            self.assertEqual(set(verified), {JOB})
            for producer in (['python3', '-m', 'other.module'],
                             ['python3', '-c', 'src.tools.record_command'],
                             ['python3', 'other.py', '-m', 'src.tools.record_command']):
                changed = copy.deepcopy(records[1])
                changed['_meta']['command'] = producer + records[1]['_meta']['command'][3:]
                (root / paths[1]).write_text(json.dumps(changed))
                with self.assertRaises(ValueError):
                    verify_evidence(root, paths, auditor, USER)

    def test_complete_chain_accepts_null_run_count_without_synthesizing_it(self):
        records, logs = fixture()
        evidence = validate_chain(records, logs, USER)
        self.assertEqual(set(evidence), {JOB})
        self.assertEqual(evidence[JOB]["allocation_gpu_hours"], 0.0)
        self.assertNotIn("run_count", evidence[JOB]["postflight_fields"])
        self.assertTrue(match_terminal(evidence[JOB]["postflight_fields"], evidence[JOB]["postflight_fields"]))
        self.assertIsNone(submit.recorded_final_hours(evidence[JOB]["postflight_fields"], 0.0, JOB))

    def test_no_missing_usage_shortcut_for_running_started_or_changed_job(self):
        for phase, key, value in ((0, "job_state", "R"), (2, "job_state", "Q"),
                                 (2, "run_count", 1), (2, "stime", "recorded start"),
                                 (2, "exec_host", "hopper-11/0"), (2, "exec_vnode", "allocated"),
                                 (2, "resources_used", {"walltime": "00:00:01"}),
                                 (2, "project", "other"), (0, "Job_Owner", "other@host")):
            records, logs = fixture()
            parsed = json.loads(logs[phase])
            parsed["Jobs"][JOB][key] = value
            logs[phase] = json.dumps(parsed)
            with self.subTest(phase=phase, key=key), self.assertRaises(ValueError):
                validate_chain(records, logs, USER)
        records, logs = fixture()
        parsed = json.loads(logs[2])
        parsed["Jobs"][JOB]["Resource_List"]["ngpus"] = 4
        logs[2] = json.dumps(parsed)
        with self.assertRaises(ValueError):
            validate_chain(records, logs, USER)

    def test_qdel_failure_wrong_ids_and_ordering_do_not_release_reservation(self):
        for mutate in (lambda r: r[1]["metrics"].update(command_exit_code=1),
                       lambda r: r[1]["_meta"].update(exit_code=1),
                       lambda r: r[1]["config"].update(command=["qdel", OTHER]),
                       lambda r: r[1]["config"].update(command=["qdel", "-W", "force", JOB]),
                       lambda r: r[2]["_meta"].update(start="2026-11-09T00:00:00"),
                       lambda r: r[1]["_meta"].update(host="different-host")):
            records, logs = fixture()
            mutate(records)
            with self.assertRaises(ValueError):
                validate_chain(records, logs, USER)
        records, logs = fixture()
        logs[2] = logs[2] + "\n" + logs[2]
        with self.assertRaises(ValueError):
            validate_chain(records, logs, USER)

    def test_sources_require_exact_logged_command_and_final_metrics(self):
        records, logs = fixture()
        auditor = SimpleNamespace(audit_file=lambda path: {"fail": [], "warn": []})
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            paths = materialize(root, records, logs)
            verified, sources = verify_evidence(root, paths, auditor, USER)
            self.assertEqual(set(verified), {JOB})
            self.assertTrue(all(row["file"] == row["path"] and row["metrics_sha256"] for row in sources))
            changed = copy.deepcopy(records[1])
            changed["config"]["command"] = ["qdel", OTHER]
            (root / paths[1]).write_text(json.dumps(changed))
            with self.assertRaises(ValueError):
                verify_evidence(root, paths, auditor, USER)
            (root / paths[1]).write_text(json.dumps(records[1]))
            (root / records[1]["_meta"]["log_path"]).write_text('FINAL_METRICS_JSON: {"command_exit_code": 1}\n')
            with self.assertRaises(ValueError):
                verify_evidence(root, paths, auditor, USER)

    def test_budget_revalidates_chain_before_accepting_zero_without_run_count(self):
        records, logs = fixture()
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, USER=USER):
            root = Path(temp).resolve()
            paths = materialize(root, records, logs)
            (root / "scripts").mkdir()
            (root / "scripts/audit_provenance.py").write_text('def audit_file(path):\n    return {"fail": [], "warn": []}\n')
            fake_auditor = SimpleNamespace(audit_file=lambda path: {"fail": [], "warn": []})
            verified, refs = verify_evidence(root, paths, fake_auditor, USER)
            fields = verified[JOB]["postflight_fields"]
            metrics = {"complete_allocation_accounting": True, "allocation_gpu_hours_by_job": {JOB: 0.0},
                       "verified_never_started_cancelled_jobs": [JOB]}
            config = {"jobs": [{"job_id": JOB, "qstat_exit": 0, "fields": fields}], "cancelled_evidence": paths}
            accounting = {"metrics": metrics, "config": config, "sources": refs,
                          "_meta": {"exit_code": 0, "command": ["python3", "src/tools/pbs_usage.py"],
                                    "log_path": "logs/runs/accounting.log",
                                    "metrics_sha256": hashlib.sha256(json.dumps(metrics, sort_keys=True).encode()).hexdigest()}}
            path = root / "results/accounting.json"
            path.write_text(json.dumps(accounting))
            (root / accounting["_meta"]["log_path"]).write_text(json.dumps(config, indent=2))
            self.assertEqual(submit._final_accounting(root, {JOB})[JOB]["gpu_hours"], 0.0)
            # Source mutation breaks authentication; the old zero claim cannot release budget.
            changed = copy.deepcopy(records[1])
            changed["config"]["command"] = ["qdel", OTHER]
            (root / paths[1]).write_text(json.dumps(changed))
            with self.assertRaises(ValueError):
                submit._final_accounting(root, {JOB})


if __name__ == "__main__":
    unittest.main()

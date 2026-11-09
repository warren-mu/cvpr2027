"""Entry/lifecycle failures use only mocks and temporary CPU files."""
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout, redirect_stderr
from unittest.mock import patch

from src.tools import smoke_e2e_views as views
from src.tools import smoke_e2e_replay as replay


class EntryTests(unittest.TestCase):
    def test_exact_command_rejects_tail_module_or_changed_identity(self):
        command = ["/scratch/env/bin/python3", "-u", "-m", views.MODULE, "--name", "smoke_e2e_fixture",
                   "--review-plan", "configs/smoke_e2e_reviewed_five.json"]
        views.exact_command(command, "smoke_e2e_fixture", "configs/smoke_e2e_reviewed_five.json")
        for changed in (["sh", *command[1:]], [*command[:3], "other.module", *command[4:]],
                        [*command, "--name", "smoke_e2e_fixture"]):
            with self.assertRaises(ValueError):
                views.exact_command(changed, "smoke_e2e_fixture", "configs/smoke_e2e_reviewed_five.json")

    def run_main(self, root, gate_code=0, row=None):
        scratch = root / "scratch"
        scratch.mkdir()
        packages = [{"pair_id": pair, "scene": "FloorPlan203", "source_config": {
            "grid_size": .25, "visibility_distance": 1.5, "field_of_view": 90}} for pair in views.APPROVED_PAIRS]
        identity = {"pbs_jobid": "123.hopper-m-02", "scratch": str(scratch), "cpu_affinity": list(range(12)),
                    "cuda_visible_devices": "GPU-11111111-2222-3333-4444-555555555555"}
        gate = {"_meta": {"exit_code": gate_code}, "metrics": {"cleanup_errors": []}}
        argv = ["fixture", "--name", "smoke_e2e_fixture", "--review-plan", "configs/p.json",
                "--timeout-seconds", "600", "--pair-timeout-seconds", "120", "--gate-timeout-seconds", "30"]
        with ExitStack() as stack, redirect_stdout(io.StringIO()) as output, redirect_stderr(io.StringIO()):
            for target, value in (("ROOT", root), ("execution_environment", lambda *_: identity),
                                  ("review_packages", lambda *_: ({}, {}, [], packages, [])),
                                  ("gate_runtime", lambda *_: identity), ("install_signals", lambda: None)):
                stack.enter_context(patch.object(views, target, value))
            stack.enter_context(patch.object(views.os, "sched_getaffinity", return_value=set(range(12)), create=True))
            stack.enter_context(patch.object(views.signal, "setitimer"))
            stack.enter_context(patch.object(views.sys, "argv", argv))
            stack.enter_context(patch.object(views, "logged_gate", return_value=(gate, {"file": "gate"}, [])))
            worker = stack.enter_context(patch.object(views, "worker_result", return_value=copy.deepcopy(row)))
            stack.enter_context(patch.object(views, "preserve_black", return_value={"fixture": True}))
            code = views.main()
        saved = json.loads((root / "results/smoke_e2e_fixture.json").read_text())
        final = json.loads(output.getvalue().split("FINAL_METRICS_JSON: ")[-1])
        self.assertEqual(saved["metrics"], final)
        return code, saved, worker.call_count

    def test_black_gate_never_starts_export(self):
        with tempfile.TemporaryDirectory() as temporary:
            code, saved, count = self.run_main(Path(temporary), gate_code=86)
        self.assertEqual((code, count), (86, 0))
        self.assertFalse(saved["metrics"]["step1_passed"])

    def test_rejected_selected_pair_stops_instead_of_filtering(self):
        row = {"pair_id": views.APPROVED_PAIRS[0], "status": "rejected", "cleanup_errors": [],
               "error": None, "rejection_reasons": ["graph_mismatch"]}
        with tempfile.TemporaryDirectory() as temporary:
            code, saved, count = self.run_main(Path(temporary), row=row)
        self.assertEqual((code, count), (1, 1))
        self.assertFalse(saved["metrics"]["step1_passed"])
        self.assertEqual(len(saved["metrics"]["exports"]), 1)
        self.assertEqual(saved["metrics"]["fresh_runs"], [])

    def test_black_plus_finalization_failure_stays_exit1(self):
        row = {"pair_id": views.APPROVED_PAIRS[0], "status": "error", "cleanup_errors": [],
               "error": None, "raw_index_error": "disk failure", "black_frame_reference": {"file": "black"}}
        with tempfile.TemporaryDirectory() as temporary:
            code, saved, count = self.run_main(Path(temporary), row=row)
        self.assertEqual((code, count), (1, 1))
        self.assertTrue(saved["metrics"]["black_frame_detected"])
        self.assertFalse(saved["metrics"]["step1_passed"])

    def test_worker_cleanup_errors_are_preserved(self):
        class Worker:
            result = {"status": "exported", "cleanup_errors": [], "error": None}
            error, identity, unity_identity = None, {"pid": 3}, {"pid": 4}
            def __init__(self, *args):
                self.result = copy.deepcopy(type(self).result)
            def receive(self):
                pass
        child = {"deadline": 1, "directory": "/scratch/test", "package": {"pair_id": "p"}}
        with patch.object(views, "Worker", Worker), patch.object(views, "stop_workers", return_value=["unknown owner"]):
            row = views.worker_result(None, child)
        self.assertEqual(row["cleanup_errors"], ["unknown owner"])
        self.assertFalse(views.completed(row, "exported"))

    def test_contract_source_requires_real_suite_command_and_nonempty_ok(self):
        command = ["python", "-m", "unittest", "discover", "-s", "tests", "-p", "test_pair_table*.py"]
        data = {"_meta": {"name": "smoke_e2e_contract", "exit_code": 0, "command": ["python", "-m",
            "src.tools.record_command", "--name", "smoke_e2e_contract", "--", *command]},
            "metrics": {"command_exit_code": 0}, "config": {"command": command}, "error": None}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            log = root / "logs/runs/smoke_e2e_contract.log"
            log.parent.mkdir(parents=True)
            log.write_text("Ran 44 tests in 1.0s\n\nOK\n")
            with patch.object(replay, "ROOT", root), patch.object(replay, "audited_result", return_value=(data, {"file": "source"})):
                self.assertEqual(replay.contract_tests("ignored"), {"file": "source"})
                log.write_text("Ran 0 tests in 0.0s\n\nOK\n")
                with self.assertRaises(ValueError):
                    replay.contract_tests("ignored")


if __name__ == "__main__":
    unittest.main()

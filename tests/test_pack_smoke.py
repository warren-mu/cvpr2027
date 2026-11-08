"""Pure synthetic fixtures for packing policy/provenance, never GPU measurement."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from src.tools import pack_smoke as pack

SPEC = importlib.util.spec_from_file_location("pack_queue", Path(__file__).resolve().parents[1] / "scripts/gpu_queue.py")
queue = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(queue)
GPUS = [f"GPU-{index:08x}-1111-2222-3333-444444444444" for index in range(4)]


class MaskTests(unittest.TestCase):
    def test_all_is_original_pbs_uuid_subset_without_mutation(self):
        environment = {"PBS_JOBID": "fixture.server", "CUDA_VISIBLE_DEVICES": ",".join(GPUS)}
        before = dict(environment)
        self.assertEqual(queue.select_gpu_ids("all", environment), GPUS)
        self.assertEqual(queue.select_gpu_ids(",".join(GPUS[2:]), environment), GPUS[2:])
        self.assertEqual(environment, before)

    def test_pbs_rejects_indices_duplicates_and_unallocated_uuid(self):
        environment = {"PBS_JOBID": "fixture.server", "CUDA_VISIBLE_DEVICES": ",".join(GPUS)}
        for request in ("0,1", f"{GPUS[0]},{GPUS[0]}", "GPU-ffffffff-1111-2222-3333-444444444444", ""):
            with self.subTest(request=request), self.assertRaises(ValueError):
                queue.select_gpu_ids(request, environment)
        with self.assertRaises(ValueError):
            queue.select_gpu_ids("all", {"PBS_JOBID": "fixture", "CUDA_VISIBLE_DEVICES": "0,1,2,3"})

    def test_wait_observation_does_not_reap(self):
        process = mock.Mock(pid=123)
        with mock.patch.object(queue.os, "waitid", return_value=None) as waitid:
            self.assertFalse(queue.exited_without_reaping(process))
        self.assertTrue(waitid.call_args.args[2] & queue.os.WNOWAIT)
        process.poll.assert_not_called()

    def test_uncertain_identity_never_signals_process_group(self):
        process = mock.Mock(pid=123)
        process.wait.return_value = 1
        row = {"process": process, "identity": {"pid": 123}, "output": mock.Mock()}
        with mock.patch.object(queue, "identity", return_value=None), \
             mock.patch.object(queue, "exited_without_reaping", return_value=True), \
             mock.patch.object(queue.time, "sleep"), mock.patch.object(queue.os, "killpg") as kill:
            errors = queue.stop_processes([row])
        self.assertTrue(errors)
        kill.assert_not_called()


class HistoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.old_root, self.old_jobs = queue.ROOT, queue.JOBS
        queue.ROOT, queue.JOBS = self.root, self.root / "private_queue"
        queue.ensure_dirs()

    def tearDown(self):
        queue.ROOT, queue.JOBS = self.old_root, self.old_jobs
        self.temp.cleanup()

    def submit(self, name="fixture"):
        args = argparse.Namespace(name=name, gpus=2, priority=0, results=None, note="CPU fixture")
        queue.cmd_submit(args, ["python3", "synthetic_fixture.py"])

    def test_terminal_name_cannot_overwrite_history(self):
        self.submit()
        queue.transition("fixture", "queue", "running")
        queue.transition("fixture", "running", "failed", exit_code=1)
        before = queue.jpath("failed", "fixture").read_bytes()
        with self.assertRaises(FileExistsError):
            self.submit()
        self.assertEqual(queue.jpath("failed", "fixture").read_bytes(), before)
        self.assertEqual(len(json.loads(before)["history"]), 3)

    def test_cancel_and_retry_preserve_original_record(self):
        self.submit()
        queue.cmd_cancel(argparse.Namespace(name="fixture"))
        old = queue.jpath("cancelled", "fixture").read_bytes()
        with self.assertRaises(ValueError):
            queue.cmd_retry(argparse.Namespace(name="fixture", as_name="fixture_retry"), [])
        queue.cmd_retry(argparse.Namespace(name="fixture", as_name="fixture_retry"), ["python3", "new_fixture.py"])
        self.assertEqual(queue.jpath("cancelled", "fixture").read_bytes(), old)
        self.assertTrue(queue.jpath("queue", "fixture_retry").exists())
        with self.assertRaises(FileExistsError):
            queue.cmd_retry(argparse.Namespace(name="fixture", as_name="fixture_retry"), ["python3", "new_fixture.py"])

    def test_existing_log_blocks_submission(self):
        path = self.root / "logs/runs/fixture.log"
        path.parent.mkdir(parents=True)
        path.write_text("preserved failure")
        with self.assertRaises(FileExistsError):
            self.submit()
        self.assertEqual(path.read_text(), "preserved failure")


def pair_fixture():
    allocation = {"name": "t00b_fixture", "pbs_jobid": "fixture.server", "host": "synthetic-host",
                  "gpu_uuids": GPUS, "cuda_visible_devices": ",".join(GPUS), "source_sha256": "synthetic-source"}
    sources, jobs = [], []
    for index, suffix in enumerate(("a", "b")):
        name, devices = f"t00b_fixture_{suffix}", GPUS[index * 2:index * 2 + 2]
        start, end = (100, 500) if index == 0 else (200, 600)
        ranks = [{"rank": rank, "gpu_uuid": device, "success": True, "completed_updates": 2,
                  "first_gradient_norm": 0.25, "parameter_delta_norm": 0.01,
                  "first_gradient_average": {"consistent": True}, "allreduce_operations": 8,
                  "compute_start_monotonic_ns": start, "compute_end_monotonic_ns": end}
                 for rank, device in enumerate(devices)]
        sources.append({"config": {"name": name, "pbs_jobid": allocation["pbs_jobid"], "host": allocation["host"],
                                   "queue_job": name, "parent_cuda_visible_devices": allocation["cuda_visible_devices"],
                                   "gpu_uuids": devices, "port": 31000 + index, "source_sha256": allocation["source_sha256"],
                                   "kind": "synthetic_packing_smoke_not_research", "backend": "nccl"},
                        "metrics": {"passed": True, "ranks": ranks, "actual_gpu_uuids": devices,
                                    "compute_interval_monotonic_ns": [start, end]},
                        "_meta": {"name": name, "exit_code": 0,
                                  "command": ["python3", "-m", "src.tools.pack_smoke", "train"]}})
        jobs.append({"name": name, "exit_code": 0, "gpu_ids": devices})
    return sources, allocation, {"jobs": jobs, "exit_code": 0, "cleanup_errors": []}


class VerificationTests(unittest.TestCase):
    def test_synthetic_pair_requires_real_fields_and_overlap(self):
        sources, allocation, worker = pair_fixture()
        errors, overlap = pack.verify_pair(sources, allocation, worker)
        self.assertEqual(errors, [])
        self.assertGreater(overlap, 0)
        sources[1]["metrics"]["ranks"][0]["first_gradient_norm"] = 0
        self.assertTrue(pack.verify_pair(sources, allocation, worker)[0])

    def test_overlapping_gpu_assignment_rejected(self):
        sources, allocation, worker = pair_fixture()
        sources[1]["metrics"]["ranks"][0]["gpu_uuid"] = GPUS[0]
        self.assertTrue(pack.verify_pair(sources, allocation, worker)[0])

    def test_nonoverlap_ports_host_and_claimed_intervals_rejected(self):
        for mutate in (
            lambda sources: sources[1]["config"].update(port=sources[0]["config"]["port"]),
            lambda sources: sources[1]["config"].update(host="other-host"),
            lambda sources: sources[1]["metrics"].update(compute_interval_monotonic_ns=[1, 2]),
            lambda sources: [row.update(compute_start_monotonic_ns=700, compute_end_monotonic_ns=800)
                             for row in sources[1]["metrics"]["ranks"]],
        ):
            sources, allocation, worker = pair_fixture()
            mutate(sources)
            self.assertTrue(pack.verify_pair(sources, allocation, worker)[0])

    def test_failed_child_and_missing_child_remain_failed(self):
        sources, allocation, worker = pair_fixture()
        sources[0]["_meta"]["exit_code"] = 124
        self.assertTrue(pack.verify_pair(sources, allocation, worker)[0])
        self.assertTrue(pack.verify_pair(sources[:1], allocation, worker)[0])

    def test_logged_hash_and_final_must_agree(self):
        with tempfile.TemporaryDirectory() as temp, mock.patch.object(pack, "ROOT", Path(temp)):
            root = Path(temp)
            metrics = {"fixture": True}
            meta = {"producer": "scripts/run_logged.py", "metrics_sha256": pack.metrics_digest(metrics),
                    "git_commit": "a" * 40, "command": ["python3", "fixture.py"], "name": "fixture",
                    "log_path": "logs/runs/fixture.log", "exit_code": 0}
            pack.save_new(root / "results/fixture.json", {"metrics": metrics, "config": {}, "_meta": meta})
            pack.save_new(root / "logs/runs/fixture.meta.json", meta)
            log = root / "logs/runs/fixture.log"
            log.write_text('FINAL_METRICS_JSON: {"fixture": true}\n# exit_code: 0\n')
            self.assertEqual(pack.load_logged(root / "results/fixture.json")[0]["metrics"], metrics)
            log.write_text('FINAL_METRICS_JSON: {"fixture": false}\n# exit_code: 0\n')
            with self.assertRaises(ValueError):
                pack.load_logged(root / "results/fixture.json")


if __name__ == "__main__":
    unittest.main()

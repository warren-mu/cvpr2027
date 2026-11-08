"""Synthetic CPU fixtures only; root executes after commit through run_logged."""
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.tools.render_gate import frame_observation, gate_exit_code, wire_observation
from src.tools.t00b_scale_prepare import check_gate, make_launcher
from src.tools import parse_t00b_scale_log as parser


def sha(data):
    return hashlib.sha256(data).hexdigest()


def logged_gate(path, gpus=4):
    metrics = {"passed": True}
    mask = ",".join(f"GPU-fixture-{index}" for index in range(gpus))
    data = {"metrics": metrics, "config": {"resolution": 255, "allocation": {
        "pbs_jobid": "fixture.pbs", "cuda_visible_devices": mask,
        "devices": [{"logical_index": index, "gpu_uuid": f"GPU-fixture-{index}"} for index in range(gpus)]}},
        "_meta": {"producer": "scripts/run_logged.py", "exit_code": 0,
                  "metrics_sha256": sha(json.dumps(metrics, sort_keys=True).encode())}}
    path.write_text(json.dumps(data))
    return data, mask


class GateFixtures(unittest.TestCase):
    def test_black_is_distinct_from_missing_shape_or_cleanup_error(self):
        capture = {"valid_shape": True, "black": False, "action_success": True}
        row = {"observation": {"valid": True, "captures": [capture, dict(capture)]}, "worker_error": None}
        rows = [copy.deepcopy(row) for _ in range(8)]
        self.assertEqual(gate_exit_code(rows, 8, []), 0)
        rows[3]["observation"]["captures"][1]["black"] = True
        rows[3]["observation"]["valid"] = False
        self.assertEqual(gate_exit_code(rows, 8, []), 86)
        self.assertEqual(gate_exit_code(rows, 8, ["cleanup uncertain"]), 1)
        rows[3]["observation"]["captures"][1]["valid_shape"] = False
        self.assertEqual(gate_exit_code(rows, 8, []), 1)
        self.assertEqual(gate_exit_code(rows[:2], 8, []), 1)

    def test_rgb_black_does_not_accept_alpha_or_missing_pixels(self):
        import numpy as np
        black = np.zeros((224, 224, 3), dtype=np.uint8)
        self.assertTrue(frame_observation(black, 224)["black"])
        rgba = np.zeros((224, 224, 4), dtype=np.uint8)
        rgba[..., 3] = 255
        self.assertFalse(frame_observation(rgba, 224)["valid_shape"])
        black[1, 2, 0] = 1
        self.assertTrue(frame_observation(black, 224)["valid"])
        self.assertFalse(frame_observation(black, 255)["valid_shape"])
        self.assertFalse(frame_observation(None, 224)["black"])

    def test_new_equal_bytes_are_frames_but_cached_object_is_not(self):
        previous = {"image": [b"same RGB bytes"]}
        metadata = {"agents": [{"screenWidth": 255, "screenHeight": 255}]}
        cached = wire_observation(previous, metadata, previous, "Pass", 3)
        new = wire_observation(previous, metadata, {"image": previous["image"]}, "Pass", 4)
        self.assertEqual(cached["fresh_main_rgb_payloads"], 0)
        self.assertTrue(cached["cached_metadata"])
        self.assertEqual(new["fresh_main_rgb_payloads"], 1)
        self.assertEqual(new["screen_width"], 255)
        self.assertEqual(new["payload_sha256"], [sha(b"same RGB bytes")])

    def test_gate_is_tied_to_logged_job_mask_topology_and_pixels(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "gate.json"
            data, mask = logged_gate(path)
            self.assertEqual(len(check_gate(path, "fixture.pbs", 4, mask)[0]["devices"]), 4)
            for job, count, allocation in (("other.pbs", 4, mask), ("fixture.pbs", 8, mask),
                                           ("fixture.pbs", 4, mask + ",outside")):
                with self.assertRaises(RuntimeError):
                    check_gate(path, job, count, allocation)
            data["metrics"]["passed"] = False
            path.write_text(json.dumps(data))
            with self.assertRaises(RuntimeError):
                check_gate(path, "fixture.pbs", 4, mask)


class LauncherFixtures(unittest.TestCase):
    SOURCE = b"""trainer.n_gpus_per_node=4
actor_rollout_ref.rollout.tensor_model_parallel_size=4
trainer.total_training_steps=300
actor_rollout_ref.rollout.gpu_memory_utilization=0.1
critic.model.fsdp_config.param_offload=False
critic.model.fsdp_config.optimizer_offload=False
data.train_batch_size=128
actor_rollout_ref.model.path=Qwen/Qwen2.5-VL-3B-Instruct
"""

    def test_topology_and_only_variant_utilization_change(self):
        for gpus, tp in ((2, 1), (4, 4), (8, 4)):
            base, steps, changes = make_launcher(self.SOURCE, "baseline", gpus)
            variant, variant_steps, variant_changes = make_launcher(self.SOURCE, "vllm04", gpus)
            self.assertEqual((steps, variant_steps), (21, 6))
            self.assertIn(f"trainer.n_gpus_per_node={gpus}\n", base)
            self.assertIn(f"tensor_model_parallel_size={tp}\n", base)
            self.assertEqual(base.replace("total_training_steps=21", "total_training_steps=6")
                             .replace("gpu_memory_utilization=0.1", "gpu_memory_utilization=0.4"), variant)
            self.assertEqual(set(variant_changes) - set(changes),
                             {"actor_rollout_ref.rollout.gpu_memory_utilization=0.1"})
            self.assertIn("critic.model.fsdp_config.param_offload=False", base)
            self.assertIn("critic.model.fsdp_config.optimizer_offload=False", base)
            self.assertIn("data.train_batch_size=128", base)
            self.assertIn("model.path=Qwen/Qwen2.5-VL-3B-Instruct", base)
        self.assertEqual(set(make_launcher(self.SOURCE, "baseline", 4)[2]), {"trainer.total_training_steps=300"})

    def test_duplicate_anchor_or_invalid_topology_fails_closed(self):
        with self.assertRaises(ValueError):
            make_launcher(self.SOURCE + b"trainer.total_training_steps=300\n", "baseline", 8)
        for count in (1, 3, 16, True):
            with self.assertRaises(ValueError):
                make_launcher(self.SOURCE, "baseline", count)


class ParserFixtures(unittest.TestCase):
    def fixture(self, directory, missing_gpu=False):
        path = lambda name: directory / name
        gate, mask = logged_gate(path("gate.json"), 8)
        config = {"allocation_gpu_count": 8, "required_completed_updates": 1,
                  "allocation": gate["config"]["allocation"], "pbs_jobid": "fixture.pbs",
                  "hook_sha256": "hook", "render_gate": {"path": str(path("gate.json")),
                                                         "sha256": sha(path("gate.json").read_bytes())}}
        path("config.json").write_text(json.dumps(config))
        path("train.log").write_text("step:1 " + " ".join(f"timing_s/{key}:2" for key in parser.REQUIRED_TIMINGS) + "\n")
        scope0 = {"phase": "initialization", "training_step": None, "validation_step": None}
        scope1 = {"phase": "training", "training_step": 1, "validation_step": None}
        scope2 = {"phase": "idle", "training_step": None, "validation_step": None}
        events = [
            {"kind": "session_start", "hook_sha256": "hook", "scope": scope0},
            {"kind": "phase", "operation": "train_begin", "step": 1, "previous_scope": scope0, "scope": scope1},
            {"kind": "rgb", "scope": scope1, "active_requests": 1, "stream_id": "stream",
             "payload_bytes": 12, "payload_sha256": "a" * 64, "field_type": "RGB_IMAGE",
             "gpu_identity": {"gpu_uuid": "GPU-fixture-0"}, "rgb_shape_valid": True,
             "rgb_valid": True, "rgb_any_nonzero": True},
            {"kind": "phase", "operation": "train_end", "step": 1, "previous_scope": scope1, "scope": scope2}]
        path("frame_events.jsonl").write_text("".join(json.dumps({**event, "server_pid": 1, "event_id": i}) + "\n"
                                                     for i, event in enumerate(events, 1)))
        gpus = {f"GPU-fixture-{index}": {"memory_used_mib": index + 10, "utilization_percent": 50}
                for index in range(7 if missing_gpu else 8)}
        resources = [{"type": "sample", "gpus": gpus, "rss_bytes": 100, "cpu_percent": 50},
                     {"type": "completion", "returncode": 0, "elapsed_seconds": 3, "cleanup_remaining": []}]
        path("resources.jsonl").write_text("".join(json.dumps(row) + "\n" for row in resources))
        files = {name: {"complete": True, "source_sha256": sha(path(name).read_bytes()),
                        "source_bytes": path(name).stat().st_size}
                 for name in ("train.log", "resources.jsonl", "frame_events.jsonl")}
        path("evidence.json").write_text(json.dumps({"complete": True, "files": files}))
        argv = ["parser", "--train-log", str(path("train.log")), "--resource-log", str(path("resources.jsonl")),
                "--frame-events", str(path("frame_events.jsonl")), "--manifest", str(path("config.json")),
                "--archive-index", str(path("evidence.json")), "--exit-code", "0", "--stage", "completed",
                "--output", str(path("result.json"))]
        with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
            code = parser.main()
        return code, json.loads(path("result.json").read_text())

    def test_eight_gpu_sampling_and_allocation_estimates(self):
        with tempfile.TemporaryDirectory() as temp:
            code, result = self.fixture(Path(temp))
            self.assertEqual(code, 0)
            self.assertTrue(result["acceptance_passed"])
            metrics = result["metrics"]
            self.assertEqual(len(metrics["gpu_resources_by_uuid"]), 8)
            self.assertEqual(metrics["actual_new_rgb_frames_per_training_step"], {"1": 1})
            estimate = metrics["full_training_estimates"]["100"]
            self.assertAlmostEqual(estimate["estimated_gpu_hours"], 100 * 2 / 3600 * 8)
            self.assertEqual(estimate["allocation_gpus"], 8)

    def test_missing_one_gpu_rejects_acceptance(self):
        with tempfile.TemporaryDirectory() as temp:
            code, result = self.fixture(Path(temp), missing_gpu=True)
            self.assertEqual(code, 1)
            self.assertFalse(result["acceptance_passed"])
            self.assertIsNone(result["metrics"]["gpu_resources_by_uuid"]["GPU-fixture-7"]["memory_peak_mib_sampled"])


if __name__ == "__main__":
    unittest.main()

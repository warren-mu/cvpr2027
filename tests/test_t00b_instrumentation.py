"""CPU-only synthetic fixtures; execute after commit through run_logged.

No simulator, GPU, training, scheduler, or research-result aggregation is run.
The installed FIFO source test skips if pinned ai2thor is absent.
"""
import ast
from collections import defaultdict, deque
import copy
from enum import IntEnum, unique
import hashlib
import gzip
import importlib.util
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
from typing import Optional
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/tools"))
from parse_t00b_train_log import frame_summary, read_steps
from t00b_prepare import HASHES, check_baseline, instrument, make_launcher, overlay_tree
from t00b_archive import archive_file, MAX_GZIP_BYTES
from t00b_runtime_hooks import phase_token, ray_environment, rgb_wire_observation, packet_dimensions


class FrameFixture:
    def __init__(self):
        self.scope = {"phase": "initialization", "training_step": None, "validation_step": None}
        self.rows = []
        self.add("session_start", hook_sha256="fixture")

    def add(self, kind, **extra):
        row = {"event_id": len(self.rows) + 1, "kind": kind, "server_pid": 777,
               "scope": dict(self.scope), **extra}
        self.rows.append(row)

    def phase(self, operation, step, new_scope):
        previous = self.scope
        self.scope = new_scope
        self.add("phase", operation=operation, step=step, previous_scope=previous, failed=False)

    def packet(self, cached=False):
        self.add("cached_metadata" if cached else "rgb",
                 field_type="METADATA_PATCH" if cached else "RGB_IMAGE",
                 active_requests=1, stream_id="controller-1", payload_bytes=12,
                 payload_sha256="0" * 64, gpu_identity={"gpu_uuid": "GPU-fixture"},
                 rgb_shape_valid=True, rgb_valid=True, rgb_any_nonzero=True)


class T00bInstrumentationTests(unittest.TestCase):
    def test_black_rgb_cannot_hide_behind_nonzero_alpha(self):
        black_rgba = bytes([0, 0, 0, 255]) * 4
        self.assertFalse(rgb_wire_observation(black_rgba, 2, 2)["rgb_valid"])
        visible_rgba = bytes([1, 0, 0, 255]) + bytes([0, 0, 0, 255]) * 3
        self.assertTrue(rgb_wire_observation(visible_rgba, 2, 2)["rgb_valid"])
        self.assertFalse(rgb_wire_observation(b"", 2, 2)["rgb_valid"])
        self.assertFalse(rgb_wire_observation(b"bad", 2, 2)["rgb_valid"])

    def test_current_packet_dimensions_allow_startup_resolution(self):
        self.assertEqual(packet_dimensions({"agents": [{"screenHeight": 300, "screenWidth": 300}]}), (300, 300))
        self.assertEqual(packet_dimensions({"agents": [{"screenHeight": 255, "screenWidth": 255}]}), (255, 255))
        self.assertIsNone(packet_dimensions(None))

    def test_phase_token_stays_in_private_file_not_ray_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "phase_token"
            path.write_text("a" * 48 + "\n")
            path.chmod(0o600)
            with patch.dict(os.environ, {"T00B_PHASE_TOKEN_FILE": str(path),
                                         "T00B_PHASE_TOKEN": "must-not-export",
                                         "T00B_OTHER_SECRET": "must-not-export-either",
                                         "T00B_SERVER_URL": "http://127.0.0.1:5000",
                                         "PYTHONPATH": "fixture"}, clear=True):
                self.assertEqual(phase_token(), "a" * 48)
                exported = ray_environment()
                self.assertEqual(set(exported), {"T00B_PHASE_TOKEN_FILE", "T00B_SERVER_URL", "PYTHONPATH"})
                self.assertNotIn("a" * 48, json.dumps(exported))
                self.assertNotIn("must-not-export", json.dumps(exported))
                path.chmod(0o644)
                with self.assertRaises(RuntimeError):
                    phase_token()

    def test_artifact_shards_roundtrip_without_deleting_raw(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = b"first\nsecond\n" + bytes(range(256)) * 3
            source = root / "raw.log"
            source.write_bytes(payload)
            record = archive_file(source, root / "repo", root / "scratch", chunk_bytes=37)
            restored = b"".join(gzip.decompress(Path(item["path"]).read_bytes()) for item in record["shards"])
            self.assertEqual(restored, payload)
            self.assertEqual(source.read_bytes(), payload)
            self.assertEqual(record["source_sha256"], hashlib.sha256(payload).hexdigest())
            self.assertEqual(record["source_bytes"], len(payload))
            self.assertEqual(record["shard_count"], len(record["shards"]))
            offset = 0
            for item in record["shards"]:
                self.assertEqual(item["raw_offset_bytes"], offset)
                offset += item["raw_bytes"]
                compressed = Path(item["path"]).read_bytes()
                self.assertLessEqual(len(compressed), MAX_GZIP_BYTES)
                self.assertEqual(item["gzip_sha256"], hashlib.sha256(compressed).hexdigest())

    def fixture(self):
        f = FrameFixture()
        f.packet()  # Initialization is measured, not attributed to a train step.
        train = {"phase": "training", "training_step": 1, "validation_step": None}
        f.phase("train_begin", 1, train)
        f.packet()
        f.packet(cached=True)
        f.phase("validation_begin", 1, {"phase": "validation", "training_step": 1, "validation_step": 1})
        f.packet()
        f.phase("validation_end", 1, train)
        f.phase("train_end", 1, {"phase": "idle", "training_step": None, "validation_step": None})
        return f

    def test_frame_cache_and_validation_attribution(self):
        result = frame_summary(self.fixture().rows, {1: {}}, "fixture", {"GPU-fixture"})
        self.assertTrue(result["valid"], result["errors"])
        self.assertEqual(result["new_rgb_packet_count_all_phases"], 3)
        self.assertEqual(result["cached_metadata_response_count"], 1)
        self.assertEqual(result["steps"]["1"]["training_main_rgb_frames"], 1)
        self.assertEqual(result["steps"]["1"]["validation_main_rgb_frames_within_step"], 1)
        self.assertEqual(result["steps"]["1"]["total_new_rgb_frames"], 2)

    def test_dropped_event_wrong_scope_or_wrong_gpu_fails(self):
        original = self.fixture().rows
        variants = [original[:3] + original[4:], copy.deepcopy(original), copy.deepcopy(original)]
        variants[1][3]["scope"]["training_step"] = 2
        variants[2][3]["gpu_identity"]["gpu_uuid"] = "GPU-other"
        for rows in variants:
            self.assertFalse(frame_summary(rows, {1: {}}, "fixture", {"GPU-fixture"})["valid"])

    def test_console_step_needs_two_phase_boundaries(self):
        result = frame_summary(self.fixture().rows[:-1], {1: {}}, "fixture", {"GPU-fixture"})
        self.assertFalse(result["valid"])
        self.assertFalse(result["steps"]["1"]["phase_end_observed"])

    def test_only_completed_console_records_count_and_all_timings_survive(self):
        text = "global_steps: 1\nstep:1 - timing_s/gen:4 - timing_s/ref:2 - timing_s/old_log_prob:1 - timing_s/values:1 - timing_s/update_critic:3 - timing_s/update_actor:3 - timing_s/step:15\nglobal_steps: 2\n"
        steps = read_steps(text)
        self.assertEqual(set(steps), {1})
        self.assertEqual(set(steps[1]["timing_s"]), {"gen", "ref", "old_log_prob", "values", "update_critic", "update_actor", "step"})

    def test_launcher_only_authorized_changes(self):
        source = b"trainer.n_gpus_per_node=4\nactor_rollout_ref.rollout.tensor_model_parallel_size=4\ntrainer.total_training_steps=300\nactor_rollout_ref.rollout.gpu_memory_utilization=0.1\ncritic.model.fsdp_config.param_offload=False\ncritic.model.fsdp_config.optimizer_offload=False\ndata.train_batch_size=128\n"
        baseline, steps, changes = make_launcher(source, "baseline")
        variant, variant_steps, variant_changes = make_launcher(source, "vllm04")
        self.assertEqual((steps, variant_steps), (21, 6))
        self.assertEqual((len(changes), len(variant_changes)), (3, 4))
        self.assertEqual(variant.replace("trainer.total_training_steps=6", "trainer.total_training_steps=21").replace("gpu_memory_utilization=0.4", "gpu_memory_utilization=0.1"), baseline)
        self.assertIn("critic.model.fsdp_config.param_offload=False", baseline)
        self.assertIn("critic.model.fsdp_config.optimizer_offload=False", variant)
        with self.assertRaises(ValueError):
            make_launcher(source, "critic_offload")
        with self.assertRaises(ValueError):
            make_launcher(source + b"trainer.n_gpus_per_node=4\n", "baseline")

    def test_overlay_preserves_source_and_relative_assets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, target = root / "source", root / "overlay"
            (source / "sub").mkdir(parents=True)
            (source / "sub/module.py").write_text("ORIGINAL")
            (source / "sub/data.json").write_text("{}")
            (source / "__init__.py").write_text("")
            overlay_tree(source, target, {"pkg/sub/module.py": b"PATCHED"}, "pkg")
            self.assertEqual((source / "sub/module.py").read_text(), "ORIGINAL")
            self.assertEqual((target / "sub/module.py").read_text(), "PATCHED")
            self.assertTrue((target / "sub/data.json").is_symlink())
            with self.assertRaises(FileExistsError):
                overlay_tree(source, target, {}, "pkg")

    def test_variant_baseline_gate_rejects_failure_or_other_allocation(self):
        metrics = {"completed_training_steps": 20}
        good = {"metrics": metrics, "acceptance_passed": True,
                "config": {"variant": "baseline", "pbs_jobid": "job-1"},
                "_meta": {"producer": "scripts/run_logged.py", "exit_code": 0,
                          "metrics_sha256": hashlib.sha256(json.dumps(metrics, sort_keys=True).encode()).hexdigest()}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "baseline.json"
            path.write_text(json.dumps(good))
            self.assertEqual(check_baseline(path, "job-1")["path"], str(path))
            with self.assertRaises(RuntimeError):
                check_baseline(path, "job-2")
            good["_meta"]["exit_code"] = 124
            path.write_text(json.dumps(good))
            with self.assertRaises(RuntimeError):
                check_baseline(path, "job-1")

    def test_pinned_fifo_cached_bytes_do_not_count_as_new_rgb(self):
        spec = importlib.util.find_spec("ai2thor")
        if spec is None or spec.origin is None:
            self.skipTest("Pinned ai2thor source not installed in this CPU environment")
        path = Path(spec.origin).parent / "fifo_server.py"
        original = path.read_bytes()
        self.assertEqual(hashlib.sha256(original).hexdigest(), HASHES["ai2thor/fifo_server.py"])
        tree = ast.parse(instrument("ai2thor/fifo_server.py", original))
        enum = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "FieldType")
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "FifoServer")
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_recv_message")
        called = []
        namespace = {"IntEnum": IntEnum, "unique": unique, "Optional": Optional,
                     "defaultdict": defaultdict, "struct": struct,
                     "msgpack": SimpleNamespace(loads=lambda body, **kwargs: json.loads(body)),
                     "_t00b_fifo_packet": lambda server, field, body, metadata=None: called.append(field)}
        exec(compile(ast.Module(body=[enum, method], type_ignores=[]), "<isolated FIFO fixture>", "exec"), namespace)
        fields = namespace["FieldType"]
        parts = deque()
        def message(records):
            for kind, body in records:
                parts.extend([struct.pack("!BI", kind.value, len(body)), body])
            parts.append(struct.pack("!BI", fields.END_OF_MESSAGE.value, 0))
        metadata = json.dumps({"agents": [{}], "thirdPartyCameras": [], "sequenceId": 1}).encode()
        image = b"same RGB bytes"
        message([(fields.METADATA, metadata), (fields.RGB_IMAGE, image)])
        message([(fields.METADATA_PATCH, json.dumps({"agentId": 0}).encode())])
        message([(fields.METADATA, metadata), (fields.RGB_IMAGE, image)])
        fake = SimpleNamespace(server_pipe=object(), timeout=10, header_size=5, header_format="!BI",
            field_types={entry.value: entry for entry in fields}, image_fields={fields.RGB_IMAGE},
            form_field_map={fields.RGB_IMAGE: "image"}, raw_metadata=None, raw_files=None, sequence_id=1,
            _read_with_timeout=lambda **kwargs: parts.popleft())
        receive = namespace["_recv_message"]
        _, first = receive(fake, None)
        _, reused = receive(fake, None)
        self.assertIs(reused, first)
        self.assertEqual(called, ["RGB_IMAGE", "METADATA_PATCH"])
        receive(fake, None)
        # Identical pixels received in a NEW wire image are another new render.
        self.assertEqual(called, ["RGB_IMAGE", "METADATA_PATCH", "RGB_IMAGE"])

    def test_pinned_vagen_source_anchors_and_launcher(self):
        source = ROOT / "third_party/VAGEN"
        if not source.exists():
            self.skipTest("Pinned VAGEN source is absent in this CPU environment")
        for relative, expected in HASHES.items():
            if relative.startswith("ai2thor/"):
                continue
            original = (source / relative).read_bytes()
            self.assertEqual(hashlib.sha256(original).hexdigest(), expected, relative)
            if relative.endswith(".py"):
                # Pure source transformation and parsing; no upstream import.
                ast.parse(instrument(relative, original))
            elif relative.endswith("run.sh"):
                for variant in ("baseline", "vllm04"):
                    make_launcher(original, variant)


if __name__ == "__main__":
    unittest.main()

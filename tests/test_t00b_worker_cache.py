"""Synthetic stdlib fixtures; execute only after commit through run_logged."""
import ast
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from src.tools.t00b_compile_cache import (
    cache_environment, freeze_worker_hook, prepare_environment,
    validate_frozen_worker_hook, worker_manifest_evidence,
)
from src.tools.t00b_worker_cache import configure_worker_cache
from src.tools.t00b_prepare import instrument as instrument_two
from src.tools.t00b_scale_prepare import instrument as instrument_scale


class WorkerCacheTests(unittest.TestCase):
    def fixture(self, directory):
        scratch = Path(directory).resolve()
        work = scratch / "run"
        work.mkdir()
        overlay, logs = work / "overlay", scratch / "evidence"
        overlay.mkdir()
        logs.mkdir()
        frozen = freeze_worker_hook(work, overlay, logs, scratch, "fixture.pbs", "fixture_run")
        manifest = {"name": "fixture_run", "pbs_jobid": "fixture.pbs",
                    "working_directory": str(work), "worker_compilation_cache": frozen,
                    "compilation_cache_environment": cache_environment(work)}
        inherited = {"PBS_JOBID": "fixture.pbs", "CVPR_SCRATCH": str(scratch),
                     "HOME": "/unchanged/home", "CUDA_VISIBLE_DEVICES": "GPU-A,GPU-B",
                     "HF_HOME": str(scratch / "models"), "UNRELATED": "unchanged",
                     "TRITON_CACHE_DIR": str(scratch / "old/triton"),
                     "TORCHINDUCTOR_CACHE_DIR": str(scratch / "old/inductor")}
        driver, _ = prepare_environment(manifest, inherited)
        return manifest, driver

    def worker(self, manifest, inherited, pid=101, start="123"):
        config = manifest["worker_compilation_cache"]
        return configure_worker_cache(config, inherited, pid=pid, hostname="fixture-host",
                                      start_ticks=start, config_sha256=config["hook_config_sha256"])

    def test_only_two_keys_change_preserving_driver_and_old_cache_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            manifest, driver = self.fixture(temp)
            before = dict(driver)
            marker = Path(driver["TRITON_CACHE_DIR"]) / "preserved.cubin"
            marker.write_bytes(b"old evidence")
            child, record = self.worker(manifest, driver)
            self.assertEqual(driver, before)
            self.assertEqual({key for key in child if child[key] != before[key]},
                             {"TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR"})
            self.assertEqual(marker.read_bytes(), b"old evidence")
            self.assertEqual(record["previous_environment"], {key: before[key] for key in record["environment"]})
            self.assertFalse(record["compiler_modes_changed"])
            self.assertEqual(Path(record["manifest_file"]).stat().st_mode & 0o777, 0o600)

    def test_distinct_workers_and_reused_pid_have_distinct_directories(self):
        with tempfile.TemporaryDirectory() as temp:
            manifest, driver = self.fixture(temp)
            paths = []
            for pid, start in ((101, "123"), (102, "123"), (101, "124")):
                child, _ = self.worker(manifest, driver, pid, start)
                paths.append(child["TRITON_CACHE_DIR"])
            self.assertEqual(len(set(paths)), 3)

    def test_repeated_hook_preserves_manifest_and_worker_cache(self):
        with tempfile.TemporaryDirectory() as temp:
            manifest, driver = self.fixture(temp)
            child, first = self.worker(manifest, driver)
            raw = Path(first["manifest_file"]).read_bytes()
            marker = Path(child["TRITON_CACHE_DIR"]) / "compiled.cubin"
            marker.write_bytes(b"retain")
            again, second = self.worker(manifest, child)
            self.assertEqual(again, child)
            self.assertEqual(first, second)
            self.assertEqual(Path(first["manifest_file"]).read_bytes(), raw)
            self.assertEqual(marker.read_bytes(), b"retain")
            # A normal compiler child inherits these values; a distinct Ray
            # worker receiving its parent's environment gets its own directory.
            other, _ = self.worker(manifest, dict(child), 103)
            self.assertNotEqual(other["TRITON_CACHE_DIR"], child["TRITON_CACHE_DIR"])

    def test_wrong_job_outside_or_redirected_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            manifest, driver = self.fixture(temp)
            for env in ({**driver, "PBS_JOBID": "other.pbs"},
                        {**driver, "TRITON_CACHE_DIR": "relative"},
                        {**driver, "TORCHINDUCTOR_CACHE_DIR": str(Path(temp).resolve() / "other")}):
                with self.assertRaises(ValueError):
                    self.worker(manifest, env)
            config = manifest["worker_compilation_cache"]
            base = Path(config["worker_base"])
            base.parent.joinpath("redirect").mkdir()
            base.symlink_to(base.parent / "redirect", target_is_directory=True)
            with self.assertRaises(ValueError):
                self.worker(manifest, driver)

    def test_partial_directory_is_preserved_and_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            manifest, driver = self.fixture(temp)
            partial = Path(manifest["worker_compilation_cache"]["worker_base"]) / "fixture-host/pid-101-start-123"
            partial.mkdir(parents=True)
            marker = partial / "partial.txt"
            marker.write_text("preserve")
            with self.assertRaises(FileExistsError):
                self.worker(manifest, driver)
            self.assertEqual(marker.read_text(), "preserve")

    def test_tampered_existing_worker_manifest_is_rejected_without_rewrite(self):
        with tempfile.TemporaryDirectory() as temp:
            manifest, driver = self.fixture(temp)
            child, record = self.worker(manifest, driver)
            path = Path(record["manifest_file"])
            record["hook_source_sha256"] = "0" * 64
            raw = json.dumps(record).encode()
            path.write_bytes(raw)
            with self.assertRaises(ValueError):
                self.worker(manifest, child)
            self.assertEqual(path.read_bytes(), raw)

    def test_frozen_source_and_config_hashes_and_exclusive_copies(self):
        with tempfile.TemporaryDirectory() as temp:
            manifest, _ = self.fixture(temp)
            frozen = manifest["worker_compilation_cache"]
            validate_frozen_worker_hook(manifest)
            for file_key, hash_key in (("hook_file", "hook_source_sha256"),
                                       ("hook_config_file", "hook_config_sha256")):
                raw = Path(frozen[file_key]).read_bytes()
                self.assertEqual(hashlib.sha256(raw).hexdigest(), frozen[hash_key])
            source = Path(frozen["hook_file"])
            original = source.read_bytes()
            with self.assertRaises(FileExistsError):
                freeze_worker_hook(Path(manifest["working_directory"]), source.parent,
                                   Path(temp).resolve() / "evidence", temp, "fixture.pbs", "fixture_run")
            self.assertEqual(source.read_bytes(), original)
            source.write_bytes(original + b"\n# changed fixture\n")
            with self.assertRaises(ValueError):
                validate_frozen_worker_hook(manifest)

    def test_manifest_raw_bytes_and_sha_survive_evidence_collection(self):
        with tempfile.TemporaryDirectory() as temp:
            manifest, driver = self.fixture(temp)
            _, record = self.worker(manifest, driver)
            raw = Path(record["manifest_file"]).read_bytes()
            rows = worker_manifest_evidence(manifest)
            self.assertEqual(len(rows), 1)
            # Roundtrip matches the JSONL archive path used by both monitors.
            archived = json.loads(json.dumps(rows[0]))
            self.assertEqual(archived["manifest_raw_json"].encode(), raw)
            self.assertEqual(archived["manifest_sha256"], hashlib.sha256(raw).hexdigest())

    def test_overlay_hook_is_sibling_to_env_vars_and_runs_no_compile(self):
        source = b"ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})\n"
        for instrument in (instrument_two, instrument_scale):
            tree = ast.parse(instrument("vagen/trainer/main_ppo.py", source))
            call = tree.body[1].value
            runtime = call.keywords[0].value
            self.assertEqual([node.value for node in runtime.keys], ["env_vars", "worker_process_setup_hook"])
            self.assertEqual(runtime.values[1].value, "t00b_worker_cache.setup_worker_cache")
            environment = runtime.values[0]
            self.assertEqual([node.value for node in environment.keys[:-1]],
                             ["TOKENIZERS_PARALLELISM", "NCCL_DEBUG"])
            self.assertIsNone(environment.keys[-1])
            self.assertEqual(environment.values[-1].func.id, "_t00b_ray_environment")
            with self.assertRaises(ValueError):
                instrument("vagen/trainer/main_ppo.py", b"ray.init()\n")


if __name__ == "__main__":
    unittest.main()

"""Path/environment-only fixtures; root executes after commit via run_logged."""
from pathlib import Path
import tempfile
import unittest

from src.tools.t00b_compile_cache import cache_environment, prepare_environment


class CompileCacheTests(unittest.TestCase):
    def fixture(self, scratch, name):
        work = scratch / name
        work.mkdir()
        manifest = {"name": name, "pbs_jobid": "fixture.pbs", "working_directory": str(work),
                    "compilation_cache_environment": cache_environment(work)}
        environment = {"PBS_JOBID": "fixture.pbs", "CVPR_SCRATCH": str(scratch),
                       "TRITON_CACHE_DIR": str(scratch / "old/shared"),
                       "TORCHINDUCTOR_CACHE_DIR": str(scratch / "old/inductor"),
                       "CUDA_VISIBLE_DEVICES": "GPU-A,GPU-B", "HOME": "/unchanged/home",
                       "HF_HOME": str(scratch / "models/shared"), "RAY_TMPDIR": str(scratch / "r/abcdef"),
                       "TMPDIR": str(scratch / "tmp"), "VLLM_ATTENTION_BACKEND": "XFORMERS"}
        return manifest, environment

    def test_only_generated_cache_paths_change_and_old_state_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            scratch = Path(directory).resolve()
            manifest, inherited = self.fixture(scratch, "baseline")
            old = Path(inherited["TRITON_CACHE_DIR"])
            old.mkdir(parents=True)
            marker = old / "__triton_launcher.so"
            marker.write_bytes(b"old evidence")
            before = dict(inherited)
            child, record = prepare_environment(manifest, inherited)
            self.assertEqual(inherited, before)
            self.assertEqual({key for key in child if child[key] != before[key]},
                             {"TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR"})
            self.assertEqual(marker.read_bytes(), b"old evidence")
            self.assertEqual(record["environment"], manifest["compilation_cache_environment"])
            self.assertFalse(record["compiler_modes_changed"])
            for value in record["environment"].values():
                self.assertTrue(Path(value).is_dir())
                self.assertEqual(Path(value).stat().st_mode & 0o777, 0o700)

    def test_distinct_jobs_and_variants_never_share_compiler_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            scratch = Path(directory).resolve()
            paths = []
            for name in ("job1_baseline", "job1_vllm04", "job2_baseline"):
                manifest, inherited = self.fixture(scratch, name)
                child, _ = prepare_environment(manifest, inherited)
                paths.append(set(child[key] for key in ("TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR")))
            self.assertFalse(paths[0] & paths[1])
            self.assertFalse(paths[0] & paths[2])

    def test_existing_run_cache_is_never_reused_or_removed(self):
        with tempfile.TemporaryDirectory() as directory:
            scratch = Path(directory).resolve()
            manifest, inherited = self.fixture(scratch, "baseline")
            child, _ = prepare_environment(manifest, inherited)
            marker = Path(child["TRITON_CACHE_DIR"]) / "preserve.so"
            marker.write_bytes(b"retained")
            with self.assertRaises(FileExistsError):
                prepare_environment(manifest, inherited)
            self.assertEqual(marker.read_bytes(), b"retained")

    def test_wrong_job_outside_scratch_and_manifest_path_drift_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            scratch = Path(directory).resolve()
            manifest, inherited = self.fixture(scratch, "baseline")
            for changed in ({**manifest, "pbs_jobid": "other.pbs"},
                            {**manifest, "working_directory": str(scratch.parent / "outside")},
                            {**manifest, "compilation_cache_environment": {}}):
                with self.assertRaises(ValueError):
                    prepare_environment(changed, inherited)
            self.assertFalse((scratch / "baseline/compile_cache").exists())
            with self.assertRaises(ValueError):
                cache_environment("relative/work")


if __name__ == "__main__":
    unittest.main()

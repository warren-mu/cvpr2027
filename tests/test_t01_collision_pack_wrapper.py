"""CPU-only shell integration fixtures; every GPU command is replaced by a stub."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
NAME = "t01_collision_pack_fixture"
STUB = r'''import hashlib
import json
import os
from pathlib import Path
import sys

argv = sys.argv[1:]
if argv[0] == "-":
    os.execv(sys.executable, [sys.executable, *argv])
assert argv[0] == "scripts/run_logged.py", argv
name = argv[argv.index("--name") + 1]
phase = ("gate" if "src.tools.render_gate" in argv else
         "probe" if "src.tools.t01_collision_probe" in argv else
         "pack" if "experiments/t00b_pack_smoke.sh" in argv else None)
assert phase is not None, argv
with Path(os.environ["FAKE_CALLS"]).open("a") as stream:
    stream.write(json.dumps({"phase": phase, "name": name, "argv": argv,
        "run_env": os.environ.get("CVPR_RUN_NAME"),
        "result_env": os.environ.get("CVPR_RESULTS_PATH")}) + "\n")
code = int(os.environ.get("FAKE_" + phase.upper() + "_CODE", "0"))
target = Path("results") / (name + ".json")
target.parent.mkdir(exist_ok=True)
if phase == "gate":
    target.write_text(json.dumps({"synthetic_cpu_fixture_only": True}))
elif phase == "probe":
    gate = Path(argv[argv.index("--gate-result") + 1])
    workers = [{"cleanup_errors": [], "black_frame_detected": False} for _ in range(4)]
    if os.environ.get("FAKE_CLEANUP_ERROR"):
        workers[0]["cleanup_errors"] = ["synthetic unknown cleanup"]
    target.write_text(json.dumps({
        "metrics": {"kind": "t01_collision_probe", "exit_code": code,
            "cleanup_errors": [], "workers": workers,
            "black_frame_detected": code == 86 or bool(os.environ.get("FAKE_UNCLEAN_BLACK"))},
        "config": {"pbs_jobid": os.environ["PBS_JOBID"],
            "gate_source": {"sha256": hashlib.sha256(gate.read_bytes()).hexdigest()}},
        "_meta": {"name": name, "exit_code": code, "end": "synthetic fixture, no GPU run"}}))
raise SystemExit(code)
'''


class ProbePackWrapperFixtures(unittest.TestCase):
    def run_fixture(self, *, gate=0, probe=0, pack=0, cleanup_error=False,
                    unclean_black=False, preexisting=False):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "experiments").mkdir()
            (root / "scripts").mkdir()
            (root / "bin").mkdir()
            (root / "scripts/scratch_env.sh").write_text("# CPU fixture; no environment installation\n")
            wrapper = root / "experiments/t01_collision_pack.sh"
            shutil.copyfile(ROOT / "experiments/t01_collision_pack.sh", wrapper)
            executable = root / "bin/python3"
            executable.write_text("#!" + sys.executable + "\n" + STUB)
            executable.chmod(0o700)
            calls_path = root / "calls.jsonl"
            env = dict(os.environ, PATH=str(root / "bin") + os.pathsep + os.environ.get("PATH", ""),
                       PBS_JOBID="123.hopper-m-02", FAKE_CALLS=str(calls_path),
                       FAKE_GATE_CODE=str(gate), FAKE_PROBE_CODE=str(probe), FAKE_PACK_CODE=str(pack))
            for key in ("FAKE_CLEANUP_ERROR", "FAKE_UNCLEAN_BLACK"):
                env.pop(key, None)
            if cleanup_error:
                env["FAKE_CLEANUP_ERROR"] = "1"
            if unclean_black:
                env["FAKE_UNCLEAN_BLACK"] = "1"
            old = root / "results" / (NAME + "_probe.json")
            if preexisting:
                old.parent.mkdir()
                old.write_text("preserve synthetic earlier evidence")
            result = subprocess.run(["bash", str(wrapper), "--name", NAME], cwd=root,
                                    env=env, capture_output=True, text=True, timeout=15)
            calls = [json.loads(line) for line in calls_path.read_text().splitlines()] if calls_path.exists() else []
            if preexisting:
                self.assertEqual(old.read_text(), "preserve synthetic earlier evidence")
            for row in calls:
                self.assertEqual(row["run_env"], row["name"])
                self.assertEqual(row["result_env"], str(root / "results" / (row["name"] + ".json")))
            return result, calls

    def test_success_has_distinct_provenance_names_and_fixed_stage_order(self):
        result, calls = self.run_fixture()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual([row["phase"] for row in calls], ["gate", "probe", "pack"])
        self.assertEqual([row["name"] for row in calls],
                         [NAME + "_gate", NAME + "_probe", "t00b_" + NAME + "_pack"])
        self.assertIn("--segmentation", calls[0]["argv"])
        self.assertEqual(calls[0]["argv"][calls[0]["argv"].index("--resolution") + 1], "224")

    def test_black_gate_stops_before_probe_and_pack(self):
        result, calls = self.run_fixture(gate=86)
        self.assertEqual(result.returncode, 86)
        self.assertEqual([row["phase"] for row in calls], ["gate"])

    def test_black_probe_stops_before_pack(self):
        result, calls = self.run_fixture(probe=86)
        self.assertEqual(result.returncode, 86)
        self.assertEqual([row["phase"] for row in calls], ["gate", "probe"])

    def test_nonblack_diagnostic_failure_remains_failure_after_successful_pack(self):
        result, calls = self.run_fixture(probe=1)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual([row["phase"] for row in calls], ["gate", "probe", "pack"])

    def test_pack_black_failure_retains_dedicated_exit_code(self):
        result, calls = self.run_fixture(probe=1, pack=86)
        self.assertEqual(result.returncode, 86)
        self.assertEqual(calls[-1]["phase"], "pack")

    def test_uncertain_cleanup_or_unclean_black_never_starts_pack(self):
        for kwargs in ({"cleanup_error": True}, {"unclean_black": True}):
            with self.subTest(kwargs=kwargs):
                result, calls = self.run_fixture(probe=1, **kwargs)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual([row["phase"] for row in calls], ["gate", "probe"])

    def test_signal_exit_never_starts_a_new_stage(self):
        for code in (130, 143):
            with self.subTest(code=code):
                result, calls = self.run_fixture(probe=code)
                self.assertEqual(result.returncode, code)
                self.assertEqual([row["phase"] for row in calls], ["gate", "probe"])

    def test_preexisting_evidence_is_preserved_without_starting_commands(self):
        result, calls = self.run_fixture(preexisting=True)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()

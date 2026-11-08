"""CPU-only stable-entry contracts; no simulator or source results read."""
import copy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

from src.tools import t01v2_stable_parallel as stable
from src.tools import t01v2_parallel as original
from src.tools import phase_watch as watch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("stable_v2_fixtures", ROOT / "tests/test_t01v2_parallel.py")
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)
NAME = "pilot_pairs_v2_stable_fixture_scene"


def scene_command(unbuffered=True):
    return ["/scratch/env/bin/python3", *(["-u"] if unbuffered else []), "-m", stable.OLD_MODULE,
            "--name", NAME, "--scene", "FloorPlan1", "--shard-config", "/scratch/owned/shard_config.json",
            "--timeout-seconds", "280"]


class StableLauncherTests(unittest.TestCase):
    def test_exact_scene_only_and_argument_identity(self):
        for unbuffered in (False, True):
            command = scene_command(unbuffered)
            before = list(command)
            self.assertEqual(stable.stable_scene_command(NAME, command),
                [stable.STABLE_MODULE if arg == stable.OLD_MODULE else arg for arg in before])
            self.assertEqual(command, before)
        for command in (["python3", "-u", "-m", "src.tools.render_gate", "--name", "gate"],
                        ["python3", "-c", "pass", stable.OLD_MODULE],
                        ["python3", "-m", "other.module", stable.OLD_MODULE],
                        ["bash", "-m", stable.OLD_MODULE],
                        ["python3", "-u", "-u", "-m", stable.OLD_MODULE]):
            self.assertIs(stable.stable_scene_command(NAME, command), command)
        for name, command in (("wrong", scene_command()), (NAME, scene_command()[:-1]),
                              (NAME, scene_command() + ["--scene", "FloorPlan2"])):
            with self.assertRaises(ValueError):
                stable.stable_scene_command(name, command)

    def test_private_binding_restored_and_environment_not_rewritten(self):
        environment = {"CUDA_VISIBLE_DEVICES": "GPU-private-subset", "HOME": "unchanged"}
        directory = Path("/scratch/private")
        with patch.object(original, "launch_logged", return_value="child") as delegate:
            def main():
                self.assertEqual(original.launch_logged(NAME, scene_command(), environment, directory), "child")
                return 7
            with patch.object(original, "main", side_effect=main):
                self.assertEqual(stable.main(), 7)
            self.assertIs(original.launch_logged, delegate)
            self.assertIs(delegate.call_args.args[2], environment)
            self.assertEqual(delegate.call_args.args[1], stable.stable_scene_command(NAME, scene_command()))
            with patch.object(original, "main", side_effect=RuntimeError("synthetic failure")):
                with self.assertRaises(RuntimeError):
                    stable.main()
            self.assertIs(original.launch_logged, delegate)
        self.assertEqual(environment, {"CUDA_VISIBLE_DEVICES": "GPU-private-subset", "HOME": "unchanged"})

    def test_budget_and_watch_entry_are_versioned(self):
        command = ["bash", "experiments/t01v2_stable_pairs.sh", "--name", NAME]
        self.assertEqual(watch.entrypoint(command, NAME, ROOT), "t01v2_stable")
        self.assertEqual(fixtures.submit.budget_group("T01", NAME, command), "T01v2")
        for task, name in (("T00b", NAME), ("T01", "unbudgeted")):
            with self.assertRaises(ValueError):
                fixtures.submit.budget_group(task, name, command)

    def test_black_child_requires_exact_version_clone_flag_and_module(self):
        data, fields, outer, gate, gate_name, child, record = fixtures.black_fixture(True)
        data["command"][1] = "experiments/t01v2_stable_pairs.sh"
        outer["_meta"]["command"] = list(data["command"])
        child["_meta"]["command"][-1] = stable.STABLE_MODULE
        child["config"].update(enable_candidate_clone=True, preflight_version=stable.PREFLIGHT_VERSION)
        self.assertEqual(watch.validate_black_retry(data, fixtures.JOB, fields, outer, gate, gate_name,
            "t01v2_stable", child, record), "t01v2_stable_scene_black_capture")
        for key, value in (("enable_candidate_clone", False), ("enable_candidate_clone", None),
                           ("preflight_version", None), ("preflight_version", "old_version")):
            bad = copy.deepcopy(child)
            bad["config"][key] = value
            with self.assertRaises(ValueError):
                watch.validate_black_retry(data, fixtures.JOB, fields, outer, gate, gate_name, "t01v2_stable", bad, record)
        for module in (stable.OLD_MODULE, "src.envs.pair_v2_clone_shard"):
            bad = copy.deepcopy(child)
            bad["_meta"]["command"][-1] = module
            with self.assertRaises(ValueError):
                watch.validate_black_retry(data, fixtures.JOB, fields, outer, gate, gate_name, "t01v2_stable", bad, record)
        for kind in ("t01v2", "t01v2_clone"):
            with self.assertRaises(ValueError):
                watch.validate_black_retry(data, fixtures.JOB, fields, outer, gate, gate_name, kind, child, record)
        command = child["_meta"]["command"]
        self.assertFalse(watch.authorized_shard_command(command, v2=True))
        self.assertFalse(watch.authorized_shard_command(command, stable=True))
        self.assertFalse(watch.authorized_shard_command(command, v2=True, clone=True, stable=True))

    def test_stable_initial_gate_black_uses_same_strict_gate_contract(self):
        data, fields, outer, gate, gate_name, child, record = fixtures.black_fixture(False)
        data["command"][1] = "experiments/t01v2_stable_pairs.sh"
        outer["_meta"]["command"] = list(data["command"])
        self.assertEqual(watch.validate_black_retry(data, fixtures.JOB, fields, outer, gate, gate_name,
            "t01v2_stable"), "render_gate_black_capture")
        bad = copy.deepcopy(data)
        bad["budget_group"] = None
        with self.assertRaises(ValueError):
            watch.validate_black_retry(bad, fixtures.JOB, fields, outer, gate, gate_name, "t01v2_stable")


if __name__ == "__main__":
    unittest.main()

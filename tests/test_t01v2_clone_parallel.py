"""Versioned clone-launcher CPU fixtures; no simulator or tests run at import."""
import copy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

from src.tools import t01v2_clone_parallel as clone
from src.tools import t01v2_parallel as original
from src.tools import phase_watch as watch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("clone_v2_fixtures", ROOT / "tests/test_t01v2_parallel.py")
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)


def scene_command(name="pilot_pairs_v2_clone_fixture_scene", unbuffered=True):
    return ["/scratch/env/bin/python3", *(["-u"] if unbuffered else []), "-m", clone.OLD_MODULE,
            "--name", name, "--scene", "FloorPlan1", "--shard-config", "/scratch/owned/shard_config.json",
            "--timeout-seconds", "280"]


class CloneLauncherTests(unittest.TestCase):
    def test_only_the_exact_scene_module_changes(self):
        for unbuffered in (False, True):
            command = scene_command(unbuffered=unbuffered)
            before = list(command)
            result = clone.clone_scene_command("pilot_pairs_v2_clone_fixture_scene", command)
            self.assertEqual(command, before)
            self.assertEqual(result, [clone.CLONE_MODULE if arg == clone.OLD_MODULE else arg for arg in before])
        gate = ["python3", "-u", "-m", "src.tools.render_gate", "--name", "pilot_pairs_v2_gate",
                "--gpus", "4", "--resolution", "255", "--segmentation", "--timeout-seconds", "180"]
        self.assertIs(clone.clone_scene_command("pilot_pairs_v2_gate", gate), gate)
        for other in (["python3", "-c", "code", clone.OLD_MODULE],
                      ["python3", "-u", "-m", "another.module", "--data", clone.OLD_MODULE],
                      ["bash", "-m", clone.OLD_MODULE],
                      ["python3", "-u", "-u", "-m", clone.OLD_MODULE]):
            self.assertIs(clone.clone_scene_command("name", other), other)

    def test_scene_name_and_explicit_cli_shape_are_guarded(self):
        with self.assertRaises(ValueError):
            clone.clone_scene_command("different_run", scene_command())
        with self.assertRaises(ValueError):
            clone.clone_scene_command("pilot_pairs_v2_clone_fixture_scene", scene_command() + ["--scene", "FloorPlan2"])
        with self.assertRaises(ValueError):
            clone.clone_scene_command("pilot_pairs_v2_clone_fixture_scene", scene_command()[:-1])

    def test_private_patch_restores_after_normal_return_and_exception(self):
        command = scene_command()
        environment = {"CUDA_VISIBLE_DEVICES": "original-subset", "HOME": "untouched"}
        directory = Path("/scratch/private")
        gate = ["python3", "-m", "src.tools.render_gate", "--name", "gate"]
        with patch.object(original, "launch_logged", return_value="synthetic child") as delegate:
            def main():
                self.assertEqual(original.launch_logged("pilot_pairs_v2_clone_fixture_scene", command, environment, directory), "synthetic child")
                original.launch_logged("gate", gate, environment, directory)
                return 7
            with patch.object(original, "main", side_effect=main):
                self.assertEqual(clone.main(), 7)
            self.assertIs(original.launch_logged, delegate)
            self.assertEqual(delegate.call_args_list[0].args[1], clone.clone_scene_command("pilot_pairs_v2_clone_fixture_scene", command))
            self.assertIs(delegate.call_args_list[0].args[2], environment)
            self.assertIs(delegate.call_args_list[1].args[1], gate)
            with patch.object(original, "main", side_effect=RuntimeError("synthetic failure")):
                with self.assertRaises(RuntimeError):
                    clone.main()
            self.assertIs(original.launch_logged, delegate)
            self.assertEqual(environment, {"CUDA_VISIBLE_DEVICES": "original-subset", "HOME": "untouched"})

    def test_clone_wrapper_is_same_subbudget_and_an_explicit_watch_kind(self):
        command = ["bash", "experiments/t01v2_clone_pairs.sh", "--name", "pilot_pairs_v2_clone_fixture"]
        self.assertEqual(watch.entrypoint(command, "pilot_pairs_v2_clone_fixture", ROOT), "t01v2_clone")
        self.assertEqual(fixtures.submit.budget_group("T01", "pilot_pairs_v2_clone_fixture", command), "T01v2")
        with self.assertRaises(ValueError):
            fixtures.submit.budget_group("T01", "pilot_pairs_other", command)

    def test_clone_and_original_black_children_cannot_cross_contracts(self):
        data, fields, outer, gate, name, child, record = fixtures.black_fixture(True)
        data["command"][1] = "experiments/t01v2_clone_pairs.sh"
        outer["_meta"]["command"] = list(data["command"])
        child["_meta"]["command"][-1] = clone.CLONE_MODULE
        child["config"]["enable_candidate_clone"] = True
        self.assertEqual(watch.validate_black_retry(data, fixtures.JOB, fields, outer, gate, name,
            "t01v2_clone", child, record), "t01v2_clone_scene_black_capture")
        for flag in (False, None, "true"):
            bad = copy.deepcopy(child)
            bad["config"]["enable_candidate_clone"] = flag
            with self.assertRaises(ValueError):
                watch.validate_black_retry(data, fixtures.JOB, fields, outer, gate, name, "t01v2_clone", bad, record)
        ordinary = copy.deepcopy(child)
        ordinary["_meta"]["command"][-1] = clone.OLD_MODULE
        with self.assertRaises(ValueError):
            watch.validate_black_retry(data, fixtures.JOB, fields, outer, gate, name, "t01v2_clone", ordinary, record)
        with self.assertRaises(ValueError):
            watch.validate_black_retry(data, fixtures.JOB, fields, outer, gate, name, "t01v2", child, record)
        self.assertFalse(watch.authorized_shard_command(child["_meta"]["command"], v2=True))
        self.assertFalse(watch.authorized_shard_command(child["_meta"]["command"], clone=True))


if __name__ == "__main__":
    unittest.main()

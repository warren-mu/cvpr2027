"""Pure synthetic T00b planning/validation checks; never import AI2-THOR."""
import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.envs.thor_isolation_diagnostics import action_plan, collect, navigation_destination, path_endpoints_match, refresh


class IsolationPlanningTests(unittest.TestCase):
    def setUp(self):
        self.poses = [
            {"objectName": "z", "position": {"x": 2., "y": 1., "z": 3.}, "rotation": {"x": 0, "y": 0, "z": 0}},
            {"objectName": "a", "position": {"x": -1., "y": 1., "z": 2.}, "rotation": {"x": 0, "y": 45, "z": 0}}]
        self.agent = {"position": {"x": 0., "y": 1., "z": 0.}, "rotation": {"x": 0, "y": 90, "z": 0},
                      "horizon": 30., "standing": True}

    def test_identity_preserves_every_pose_and_agent_without_reset(self):
        actions, edit = action_plan("no_reset_identity", self.poses, self.agent)
        self.assertEqual([item[1] for item in actions], ["SetObjectPoses", "TeleportFull"])
        self.assertEqual(actions[0][2]["objectPoses"], self.poses)
        self.assertEqual(actions[1][2], self.agent)
        self.assertEqual(edit, {})
        actions[0][2]["objectPoses"][0]["position"]["x"] = 999
        self.assertEqual(self.poses[0]["position"]["x"], 2.)

    def test_edit_is_one_fixed_half_metre_translation_and_complete_undo(self):
        before = copy.deepcopy(self.poses)
        actions, edit = action_plan("no_reset_edit_undo", self.poses, self.agent)
        edited = actions[0][2]["objectPoses"]
        self.assertEqual(edit["edited_object_name"], "a")
        self.assertEqual(edited[0], before[0])
        self.assertEqual(edited[1]["position"], {"x": -0.5, "y": 1., "z": 2.})
        self.assertEqual(edited[1]["rotation"], before[1]["rotation"])
        self.assertEqual(actions[1][2]["objectPoses"], before)
        self.assertEqual(self.poses, before)
        self.assertEqual([item[1] for item in actions], ["SetObjectPoses", "SetObjectPoses"])
        self.assertIsNone(edit["physical_validity"])

    def test_revisit_keeps_heading_horizon_and_exact_return_request(self):
        target = {"x": 1., "y": 1., "z": 0.}
        actions, _ = action_plan("no_reset_revisit", self.poses, self.agent, target)
        self.assertEqual([item[1] for item in actions], ["TeleportFull", "TeleportFull"])
        self.assertEqual(actions[0][2], {**self.agent, "position": target})
        self.assertEqual(actions[1][2], self.agent)
        with self.assertRaises(ValueError):
            action_plan("no_reset_revisit", self.poses, self.agent)

    def test_path_checks_both_endpoints_and_rejects_partial_or_unknown_shape(self):
        start, target = {"x": 0., "z": 0.}, {"x": 1., "z": 0.}
        good = {"corners": [start, target], "status": "PathComplete"}
        self.assertTrue(path_endpoints_match(good, start, target, 0.01))
        self.assertTrue(path_endpoints_match({**good, "status": 0}, start, target, 0.01))
        self.assertFalse(path_endpoints_match({**good, "status": "PathPartial"}, start, target, 0.01))
        self.assertFalse(path_endpoints_match({"corners": [start]}, start, target, 0.01))
        self.assertFalse(path_endpoints_match(None, start, target, 0.01))
        for corners in ([{"x": 0.1, "z": 0.}, target], [start, {"x": 1.1, "z": 0.}]):
            self.assertFalse(path_endpoints_match({"corners": corners}, start, target, 0.01))

    def test_navigation_uses_actual_build_vector_target_and_checks_return_direction(self):
        calls = []
        def query(**kwargs):
            calls.append(kwargs)
            self.assertIn("target", kwargs)
            self.assertNotIn("x", kwargs)
            return SimpleNamespace(metadata={"lastActionSuccess": True,
                "actionReturn": {"corners": [kwargs["position"], kwargs["target"]], "status": 0}})
        target = {"x": 1., "y": 1., "z": 0.}
        raw = {}
        with patch("src.envs.thor_isolation_diagnostics.snapshot", return_value={}):
            destination = navigation_destination(SimpleNamespace(step=query), [target, target], self.agent, raw, 0.01)
        self.assertEqual(destination, target)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["position"], self.agent["position"])
        self.assertEqual(calls[1]["target"], self.agent["position"])

    def test_completion_does_not_turn_nonzero_rgb_into_pass(self):
        metrics = {"cases": [{"status": "completed", "rgb": {"rgb_exact_zero": False, "max_pixel_difference": 1}}
                              for _ in range(20)]}
        refresh(metrics, final=True)
        self.assertTrue(metrics["diagnostic_completed"])
        self.assertFalse(metrics["exact_zero_passed"])
        metrics["cases"].pop()
        refresh(metrics, final=True)
        self.assertFalse(metrics["diagnostic_completed"])
        self.assertIsNone(metrics["exact_zero_passed"])

    def test_partial_worker_startup_still_cleans_owned_first_worker(self):
        worker = SimpleNamespace(latest={"cases": [{"status": "running"}]},
                                 error=None, identity={"pid": 123}, unity_identity=None)
        with patch("src.envs.thor_isolation_diagnostics.Worker", side_effect=[worker, RuntimeError("startup failed")]), \
             patch("src.envs.thor_isolation_diagnostics.stop_workers", return_value=[]) as cleanup, \
             patch("src.envs.thor_isolation_diagnostics.traceback.print_exc"):
            outputs, error, cleanup_errors = collect(None, [{}, {}], float("inf"))
        cleanup.assert_called_once_with([worker])
        self.assertEqual(outputs[0]["observations"], worker.latest)
        self.assertIn("startup failed", error)
        self.assertEqual(cleanup_errors, [])


if __name__ == "__main__":
    unittest.main()

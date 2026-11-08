"""Synthetic supplement-2 CPU boundaries; no Unity or actual result reads."""
import copy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from src.envs import pair_v2_stable_policy as stable
from src.envs import pair_v2_policy as original
from src.envs import pair_v2_shard as shard

ROOT = Path(__file__).resolve().parents[1]
def fixture_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tests" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

policy_fixtures = fixture_module("stable_policy_cpu_fixtures", "test_pair_v2_policy.py")
shard_fixtures = fixture_module("stable_shard_cpu_fixtures", "test_pair_v2_shard.py")


def paired_views(count=250, difference=11, *, right_pixels=0):
    first = policy_fixtures.view(anchor_pixels=count)
    second = policy_fixtures.view(anchor_pixels=right_pixels)
    second["rgb"].reshape(-1, 3)[30000:30000 + count, 0] += difference
    second["marked"] = second["rgb"].copy()
    return first, second


def classification(first, second, kind="look_flip"):
    value = policy_fixtures.PairV2PolicyTests().classification(kind)
    value["evidence"] = (stable.look_evidence(first, second, "X") if kind.startswith("look")
                         else original.compare_view(first, second, ["A", "B", "X"]))
    value["anchor_pixels_w0"] = int(original._mask(first, "X").sum())
    value["anchor_pixels_w1"] = int(original._mask(second, "X").sum())
    return value


class StableEvidenceTests(unittest.TestCase):
    def test_one_world_absent_anchor_can_supply_real_250_pixel_union_evidence(self):
        first, second = paired_views()
        second["masks_by_name"].pop("X")
        measured = stable.look_evidence(first, second, "X")
        self.assertEqual((measured["anchor_pixels_w0"], measured["anchor_pixels_w1"]), (250, 0))
        self.assertEqual(measured["evidence_pixels"], 250)
        values = classification(first, second)
        self.assertFalse(stable._ORIGINAL_CLASSIFY("look_flip", **values)["acceptance_passed"])
        self.assertTrue(stable.classify_pair("look_flip", **values)["acceptance_passed"])
        values["restore_deterministic"] = False
        self.assertIn("state_restore", stable.classify_pair("look_flip", **values)["reasons"])

    def test_exact_10_difference_and_249_area_fail_while_11_and_250_pass(self):
        for count, difference, expected in ((250, 10, False), (249, 11, False), (250, 11, True)):
            first, second = paired_views(count, difference)
            measured = stable.look_evidence(first, second, "X")
            self.assertIs(measured["evidence_passed"], expected)
            self.assertIs(stable.classify_pair("look_flip", **classification(first, second))["acceptance_passed"], expected)

    def test_candidate_only_changes_cannot_count_as_look_anchor_evidence(self):
        first, second = paired_views(difference=0)
        second["rgb"][10:20, 10:35, 0] += 11
        second["marked"] = second["rgb"].copy()
        self.assertTrue(original.compare_view(first, second, ["A", "B", "X"])["evidence_passed"])
        self.assertFalse(stable.look_evidence(first, second, "X")["evidence_passed"])
        self.assertFalse(stable.classify_pair("look_flip", **classification(first, second))["acceptance_passed"])

    def test_overlapping_masks_count_once_and_public_evidence_still_required(self):
        first, second = paired_views(right_pixels=250)
        measured = stable.look_evidence(first, second, "X")
        self.assertEqual(measured["relevant_mask_union_pixels"], 250)
        second["marked"] = first["marked"].copy()
        result = stable.classify_pair("look_flip", **classification(first, second))
        self.assertFalse(result["acceptance_passed"])
        self.assertIn("policy_input_evidence", result["reasons"])

    def test_declared_union_version_anchor_and_counts_must_match(self):
        first, second = paired_views()
        for key, value in (("evidence_version", "old"), ("anchor_name", "A"),
                           ("relevant_names", ["A", "X"]), ("anchor_pixels_w1", 250),
                           ("evidence_pixels", 251)):
            values = classification(first, second)
            values["evidence"][key] = value
            self.assertFalse(stable.classify_pair("look_flip", **values)["acceptance_passed"], key)

    def test_direct_requires_both_anchor_masks_at_250(self):
        for second_count, expected in ((0, False), (249, False), (250, True)):
            first, second = paired_views(right_pixels=second_count)
            result = stable.classify_pair("direct_flip", **classification(first, second, "direct_flip"))
            self.assertIs(result["acceptance_passed"], expected)

    def test_shared_prefix_and_common_observation_witness_remain_required(self):
        first, second = paired_views()
        values = classification(first, second)
        values["prefix"]["same_input_at_t"] = False
        self.assertIn("look_input_not_actually_shared", stable.classify_pair("look_flip", **values)["reasons"])
        values = classification(first, second)
        values["witness_w1"]["replay"]["executed_actions"][0] = "turn_left"
        self.assertIn("replay_does_not_start_with_common_observation",
                      stable.classify_pair("look_flip", **values)["reasons"])

    def test_frozen_build_accepts_one_sided_anchor_only_with_private_new_hooks(self):
        class OneSided(shard_fixtures.SyntheticSimulator):
            def capture(self):
                value = super().capture()
                if self.phase == "after" and self.current["key"].endswith("_w1"):
                    value["masks_by_name"]["X"][:] = False
                return value
        sim = OneSided(look=True)
        with patch.object(shard, "choose_look", stable.choose_look), \
             patch.object(original, "classify_pair", stable.classify_pair), \
             patch.object(shard, "observation_actions", return_value=([{"kind": "turn_right"}], [])), \
             patch.object(shard, "wait_replay", side_effect=shard_fixtures.successful_replay), \
             patch.object(shard, "export_pair", return_value="synthetic_pair.json"), \
             patch.object(shard, "sha", return_value="synthetic_sha"):
            row = shard_fixtures.build(sim)
        self.assertEqual(row["status"], "passed", row)
        self.assertEqual(row["anchor_pixels"], [250, 0])
        self.assertEqual(row["evidence"]["relevant_names"], ["X"])
        self.assertEqual(row["witness_actions"], [["turn_right", "select 1"], ["turn_right", "select 2"]])
        self.assertIs(original.classify_pair, stable._ORIGINAL_CLASSIFY)


if __name__ == "__main__":
    unittest.main()

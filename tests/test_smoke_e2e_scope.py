"""CPU fixtures only. No synthetic fixture is a live cache certificate."""
from dataclasses import replace
import copy
import unittest
from unittest.mock import patch

from src.envs.paired_env import RGBFrame
from src.envs.pair_table import RenderedView, label_assignment
from src.envs.pair_table_cache import CacheSample, CACHE_POLICY, sample_state_keys, require_cache_validation
from src.envs.pair_table_export import build_table, state_id
from src.tools import smoke_e2e_scope as scope
from src.tools import smoke_e2e_replay as replay


def mask(count, offset=0):
    return bytes(offset) + bytes([1]) * count + bytes(255 * 255 - offset - count)


class Renderer:
    def __init__(self, pair_id):
        self.pair_id = pair_id
    def domain(self, world, state):
        return {"complete": True, "moves": [], "evidence": {"synthetic": True}}
    def render_pair(self, state):
        views = []
        for world in (0, 1):
            source = {"pair_id": self.pair_id, "state_id": state_id(state), "world_id": f"w{world}",
                      "snapshot_sha256": str(world) * 64, "fixture": True}
            views.append(RenderedView(RGBFrame(bytes([40 + world, 50, 60]) * (255 * 255), 255, 255),
                {"X": mask(300, 1000), "A": mask(300), "B": mask(300, 500)}, {"A": 1, "B": 1, "X": 2},
                {"position": {"x": 0, "y": 0, "z": 0}, "heading": state.heading_index},
                "cached_fixture_" + self.pair_id, f"cached:{self.pair_id}:{state_id(state)}:{world}", source, True))
        return tuple(views)


def tables():
    result = {}
    for index in range(3):
        name = f"fixture_{index}"
        table = build_table(Renderer(name), pair_id=name, scene="FloorPlan1", kind="direct_flip",
            instruction="Select the marked object closest to the kettle.", anchor_name="X",
            candidate_names=("A", "B"), target_names=("A", "B"),
            initial_position={"x": 0, "y": 0, "z": 0}, initial_heading=0, source={"synthetic": True})
        result[name] = table
    return result


def samples_for(values, plan):
    samples = []
    for pair, state in plan["selected_keys"]:
        table = values[pair]
        old = table.states[state].captures
        fresh = tuple(replace(view, process_id="fresh_fixture_" + pair,
                              render_id="fresh:" + view.render_id) for view in old)
        labels = label_assignment(pair, table.candidate_names)
        samples.append(CacheSample(pair, state, "X", ("A", "B"), old, fresh, (labels, labels), (labels, labels)))
    return samples


class SmokeScopeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tables = tables()
        cls.population = [(pair, state) for pair, table in cls.tables.items() for state in table.states]
        cls.plan = scope.sample_plan(cls.population)
        cls.samples = samples_for(cls.tables, cls.plan)

    def test_exact_twenty_repeatable_and_formal_minimum_unchanged(self):
        self.assertEqual(scope.sample_plan(reversed(self.population)), self.plan)
        self.assertEqual(len({tuple(key) for key in self.plan["selected_keys"]}), 20)
        self.assertEqual(CACHE_POLICY["minimum_distinct_pair_states"], 200)
        with self.assertRaises(ValueError):
            sample_state_keys(self.population, seed=0, sample_size=20)
        with self.assertRaises(ValueError):
            scope.sample_plan(self.population[:19])

    def test_missing_state_is_unknown_and_no_replacement_is_permitted(self):
        report = scope.validate_sample(self.samples[:-1], population=self.population, plan=self.plan)
        self.assertFalse(report["sample_complete"])
        self.assertEqual(len(report["missing_state_keys"]), 1)
        self.assertFalse(report["smoke_cache_matches_live"])
        with self.assertRaises(ValueError):
            scope.validate_sample([*self.samples, self.samples[0]], population=self.population, plan=self.plan)
        changed = copy.deepcopy(self.plan)
        changed["selected_keys"].reverse()
        with self.assertRaises(ValueError):
            scope.validate_sample(self.samples, population=self.population, plan=changed)

    def test_synthetic_pass_never_authorizes_live_or_formal_path(self):
        report = scope.validate_sample(self.samples, population=self.population, plan=self.plan)
        self.assertTrue(report["checks_passed"])
        self.assertFalse(report["smoke_cache_matches_live"])
        self.assertIsNone(report["formal_cache_matches_live"])
        with self.assertRaises(ValueError):
            scope.authorize_smoke(list(self.tables.values()), pair_checks={}, report=report, authenticated_ids=set())
        with self.assertRaises(ValueError):
            require_cache_validation(report)
        self.assertTrue(all(table.cache_verified is False for table in self.tables.values()))

    def test_one_mask_mismatch_blocks_whole_sample(self):
        first = self.samples[0]
        wrong = replace(first.fresh[0], masks={**first.fresh[0].masks, "X": mask(500, 1000)})
        changed = [replace(first, fresh=(wrong, first.fresh[1])), *self.samples[1:]]
        result = scope.validate_sample(changed, population=self.population, plan=self.plan)
        self.assertFalse(result["checks_passed"])
        self.assertFalse(result["smoke_cache_matches_live"])
        self.assertEqual(result["observed_distinct_states"], 20)

    def test_reusing_export_process_or_fresh_render_is_rejected(self):
        first = self.samples[0]
        wrong = replace(first.fresh[0], process_id=first.cached[0].process_id)
        with self.assertRaises(ValueError):
            scope.validate_sample([replace(first, fresh=(wrong, first.fresh[1])), *self.samples[1:]],
                                  population=self.population, plan=self.plan)
        second = replace(self.samples[1], fresh=first.fresh)
        with self.assertRaises(ValueError):
            scope.validate_sample([first, second, *self.samples[2:]], population=self.population, plan=self.plan)

    def test_cpu_environment_replay_keeps_formal_flag_false_and_tau0_limit(self):
        # Only the certificate boundary is mocked; actual table/env/witness
        # actions and all five-turn/reward/label semantics remain unchanged.
        certificate = {"tables": {key: table.manifest_reference for key, table in self.tables.items()}}
        packages = {key: {"t01": {"witnesses": [
            {"actions": [{"kind": "select", "target_name": "A"}]},
            {"actions": [{"kind": "select", "target_name": "B"}]}]}} for key in self.tables}
        with patch.object(scope, "authorize_smoke", return_value=certificate):
            result = replay.replay(self.tables, packages, {}, {}, set(), certificate)
        self.assertTrue(result["witness_replay_passed"])
        self.assertTrue(all(row["success"] and row["tau"] == 0 for row in result["world_replays"]))
        self.assertEqual(result["coverage"]["pre_evidence_credit_coverage"], "unmeasured")
        self.assertEqual(result["coverage"]["paired_generation_dedup_coverage"], "unmeasured")
        self.assertTrue(all(table.cache_verified is False for table in self.tables.values()))


if __name__ == "__main__":
    unittest.main()

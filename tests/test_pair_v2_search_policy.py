"""Synthetic ordering/identity checks only; no simulator or research data."""
import copy
import unittest

from src.envs.pair_v2_search_policy import (SEARCH_POLICY, RoundRobinScheduler,
    eligible_clone_sources, eligible_moved_object, excluded_clone_source_type, union_identity_sets)


def obj(name, category, **extra):
    return {"name": name, "objectType": category, "pickupable": True, **extra}


class SearchPolicyTests(unittest.TestCase):
    def test_exact_rolling_categories_and_native_variants(self):
        for category in (*SEARCH_POLICY["excluded_moved_object_types"],
                         *SEARCH_POLICY["excluded_native_variant_types"], "MugBroken"):
            self.assertTrue(excluded_clone_source_type(category), category)
        self.assertFalse(excluded_clone_source_type("PenHolder"))
        self.assertFalse(excluded_clone_source_type("Mug"))

    def test_native_break_state_does_not_need_a_renamed_category(self):
        self.assertFalse(eligible_moved_object(obj("m", "Mug", isBroken=True)))
        self.assertFalse(eligible_moved_object(obj("b", "Bread", isSliced=True)))
        self.assertTrue(eligible_moved_object(obj("m", "Mug", breakable=True, isBroken=False)))

    def test_singleton_sources_fixed_order_maximum_six_without_input_mutation(self):
        objects = [obj(str(i), category) for i, category in enumerate(
            ("Vase", "Mug", "Book", "Cup", "Bowl", "SoapBar", "Candle", "DishSponge", "Apple"))]
        original = copy.deepcopy(objects)
        first = eligible_clone_sources(objects)
        self.assertEqual([o["objectType"] for o in first], ["Book", "Bowl", "Candle", "Cup", "DishSponge", "Mug"])
        self.assertEqual(first, eligible_clone_sources(list(reversed(objects))))
        first[0]["name"] = "changed"
        self.assertEqual(objects, original)

    def test_nonpickupable_or_non_singleton_sources_are_not_retried(self):
        objects = [obj("a", "Mug"), obj("b", "Mug", pickupable=False), obj("c", "Book", pickupable=False)]
        self.assertEqual(eligible_clone_sources(objects), [])

    def test_natural_rolling_and_broken_candidates_remain_with_clone_group(self):
        objects = [obj("a", "Apple"), obj("b", "Apple", isBroken=True), obj("m", "Mug"),
                   obj("copy", "Mug"), obj("x", "Book"), obj("bad", "Pen")]
        rows = union_identity_sets(objects, {"candidate_names": ["m", "copy"]})
        self.assertEqual([(r["target_type"], r["anchor_type"], r["search_candidate_origin"]) for r in rows],
                         [("Apple", "Book", "natural"), ("Mug", "Book", "cloned")])

    def test_no_clone_still_retains_every_natural_group(self):
        objects = [obj("a", "Apple"), obj("b", "Apple"), obj("m", "Mug"), obj("n", "Mug"), obj("x", "Book")]
        rows = union_identity_sets(objects)
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["search_candidate_origin"] == "natural" for row in rows))
        self.assertEqual(rows, union_identity_sets(list(reversed(objects))))

    def test_ambiguous_or_unknown_lineage_inventory_rejected(self):
        with self.assertRaises(ValueError):
            union_identity_sets([obj("same", "Mug"), obj("same", "Book")])
        with self.assertRaises(ValueError):
            union_identity_sets([obj("m", "Mug"), obj("x", "Book")], {"candidate_names": ["m", "x"]})
        with self.assertRaises(ValueError):
            eligible_clone_sources([obj("m", "")])

    def groups(self):
        return [{"candidate_names": ["a", "b"], "anchor_name": x} for x in ("x", "y")]

    def test_round_robin_interleaves_instead_of_exhausting_first_group(self):
        scheduler = RoundRobinScheduler(self.groups())
        self.assertEqual([scheduler.next()["anchor_name"] for _ in range(6)], ["x", "y"] * 3)

    def test_ten_consecutive_physical_rejections_skip_only_that_combination(self):
        scheduler = RoundRobinScheduler(self.groups())
        for i in range(10):
            x, y = scheduler.next(), scheduler.next()
            state = scheduler.observe(x, physical_rejected=True)
            scheduler.observe(y, physical_rejected=False)
            self.assertEqual(state["skipped"], i == 9)
        self.assertEqual(scheduler.next()["anchor_name"], "y")
        scheduler.exhaust(self.groups()[1])
        self.assertIsNone(scheduler.next())

    def test_nonphysical_outcome_resets_streak_not_lifetime_failure_count(self):
        scheduler = RoundRobinScheduler(self.groups()[:1])
        group = scheduler.next()
        for _ in range(9):
            scheduler.observe(group, physical_rejected=True)
        scheduler.observe(group, physical_rejected=False)
        self.assertFalse(scheduler.observe(group, physical_rejected=True)["skipped"])
        self.assertEqual(scheduler.snapshot()[0]["physical_rejections"], 10)
        self.assertEqual(scheduler.snapshot()[0]["consecutive_physical_rejections"], 1)

    def test_empty_exhausted_duplicate_and_unobserved_outcome(self):
        self.assertIsNone(RoundRobinScheduler([]).next())
        with self.assertRaises(ValueError):
            RoundRobinScheduler(self.groups() * 2)
        scheduler = RoundRobinScheduler(self.groups())
        with self.assertRaises(ValueError):
            scheduler.observe(self.groups()[0], physical_rejected=None)
        scheduler.exhaust(self.groups()[0])
        self.assertEqual(scheduler.next()["anchor_name"], "y")


if __name__ == "__main__":
    unittest.main()

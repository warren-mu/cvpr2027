"""CPU-only synthetic contract checks; no simulator or research artifacts.

Run from the repository root with:
    python3 -B -m unittest discover -s tests -p 'test_pair_core.py' -v

These fixtures validate bookkeeping and rejection paths. They do not establish
physical validity, render equality, executable witnesses, or T01 completion.
"""
import unittest

from src.envs.pair_core import (
    Action, Mark, NavigationGraph, ObjectState, PathEvidence, Point3,
    RelationEvidence, RelationalTask, Transition, Viewpoint, ViewState,
    WorldSnapshot, bounded_witness, build_navigation_graph,
    farthest_point_sample, make_scene_split, navigation_transitions,
    resolve_target, selection_action, visible_marks,
)


def obj(object_id, object_type="Cup", *, name=None, x=0, **flags):
    return ObjectState(object_id, name or object_id, object_type,
                       Point3(x, 0, 0), Point3(0, 0, 0), **flags)


def world(*objects, key="synthetic-world", complete=True):
    return WorldSnapshot("synthetic-kitchen", key, tuple(objects), complete)


class InventoryAndSplitTests(unittest.TestCase):
    def test_full_poses_include_union_of_moveable_and_pickupable(self):
        snapshot = world(obj("movable", name="z", moveable=True, x=1),
                         obj("pickup", name="a", pickupable=True, x=2),
                         obj("both", name="m", moveable=True, pickupable=True),
                         obj("static"))
        poses = snapshot.full_object_poses()
        self.assertEqual([pose["objectName"] for pose in poses], ["a", "m", "z"])
        self.assertEqual(poses[0]["position"], {"x": 2, "y": 0, "z": 0})
        self.assertEqual(poses[-1]["rotation"], {"x": 0, "y": 0, "z": 0})
        poses[0]["position"]["x"] = 999
        self.assertEqual(snapshot.full_object_poses()[0]["position"]["x"], 2)

    def test_incomplete_or_duplicate_inventory_cannot_export_as_complete(self):
        with self.assertRaises(ValueError):
            world(obj("pickup", pickupable=True), complete=False).full_object_poses()
        with self.assertRaises(ValueError):
            world(obj("duplicate"), obj("duplicate"))
        with self.assertRaises(ValueError):
            world(obj("first", name="same"), obj("second", name="same"))

    def test_scene_split_holds_out_entire_scene_ids(self):
        rooms = {"kitchen-b": "kitchen", "living-a": "living_room",
                 "kitchen-a": "kitchen", "living-b": "living_room"}
        split = make_scene_split(rooms, ["living-b", "kitchen-b"])
        self.assertEqual(split["train_scenes"], ["kitchen-a", "living-a"])
        self.assertEqual(split["test_scenes"], ["kitchen-b", "living-b"])
        self.assertFalse(set(split["train_scenes"]) & set(split["test_scenes"]))
        self.assertEqual(set(split["train_scenes"]) | set(split["test_scenes"]), set(rooms))
        self.assertEqual(split, make_scene_split(dict(reversed(list(rooms.items()))),
                                                 ["kitchen-b", "living-b"]))

    def test_scene_split_rejects_invalid_holdout(self):
        rooms = {"train": "kitchen", "test": "living_room"}
        for held_out in ([], ["train", "test"], ["missing"], ["test", "test"]):
            with self.subTest(held_out=held_out), self.assertRaises(ValueError):
                make_scene_split(rooms, held_out)
        with self.assertRaises(ValueError):
            make_scene_split({"train": "bedroom", "test": "kitchen"}, ["test"])

    def test_farthest_sampling_is_deterministic_and_preserves_spacing(self):
        points = [Point3(i * 2, 0, 0) for i in range(16)]
        first = farthest_point_sample(points, count=12, seed=7)
        second = farthest_point_sample(list(reversed(points)) + points, count=12, seed=7)
        self.assertEqual(first, second)
        self.assertTrue(first.complete)
        for i, point in enumerate(first.points):
            for other in first.points[i + 1:]:
                self.assertGreaterEqual(point.distance(other), 1.0)

    def test_insufficient_viewpoint_spacing_is_not_padded(self):
        sample = farthest_point_sample([Point3(i / 100, 0, 0) for i in range(20)],
                                       count=12, seed=0)
        self.assertFalse(sample.complete)
        self.assertEqual(sample.reason, "insufficient_reachable_points_at_required_spacing")
        self.assertLess(len(sample.points), sample.requested_count)


class RelationTests(unittest.TestCase):
    def setUp(self):
        self.objects = (obj("cup-a"), obj("cup-b"), obj("anchor", "Kettle"))
        self.snapshot = world(*self.objects)

    def resolve(self, relation, values, *, snapshot=None, tolerance=0.0):
        snapshot = snapshot or self.snapshot
        evidence = RelationEvidence(snapshot.world_key, relation, "synthetic-explicit-facts",
                                    {(candidate, "anchor"): value for candidate, value in values.items()})
        return resolve_target(RelationalTask("Cup", relation, "Kettle"), snapshot, evidence,
                              distance_tie_tolerance_m=tolerance)

    def test_next_to_resolves_from_each_worlds_evidence(self):
        other = world(*self.objects, key="edited-world")
        self.assertEqual(self.resolve("next to", {"cup-a": 1, "cup-b": 2}).target_id, "cup-a")
        self.assertEqual(self.resolve("next to", {"cup-a": 2, "cup-b": 1},
                                      snapshot=other).target_id, "cup-b")
        wrong_world = RelationEvidence("other-world", "next to", "synthetic", {})
        with self.assertRaises(ValueError):
            resolve_target(RelationalTask("Cup", "next to", "Kettle"), self.snapshot,
                           wrong_world, distance_tie_tolerance_m=0)

    def test_nearest_ties_and_explicit_near_ties_are_rejected(self):
        for values, tolerance in (({"cup-a": 1, "cup-b": 1}, 0),
                                  ({"cup-a": 1, "cup-b": 1.01}, 0.02)):
            with self.subTest(values=values):
                result = self.resolve("next to", values, tolerance=tolerance)
                self.assertIsNone(result.target_id)
                self.assertEqual(result.reason, "target_not_unique")

    def test_unknown_evidence_is_not_assumed_false_or_far_away(self):
        for relation, value in (("next to", 1), ("in", True), ("on", True)):
            with self.subTest(relation=relation):
                result = self.resolve(relation, {"cup-a": value})
                self.assertIsNone(result.target_id)
                self.assertEqual(result.reason, "unknown_relation_evidence")

    def test_in_and_on_require_exactly_one_verified_match(self):
        for relation in ("in", "on"):
            with self.subTest(relation=relation):
                self.assertEqual(self.resolve(relation, {"cup-a": True, "cup-b": False}).target_id,
                                 "cup-a")
                for values in ({"cup-a": True, "cup-b": True},
                               {"cup-a": False, "cup-b": False}):
                    self.assertIsNone(self.resolve(relation, values).target_id)
                with self.assertRaises(ValueError):
                    self.resolve(relation, {"cup-a": 1, "cup-b": 0})

    def test_anchor_uniqueness_and_complete_inventory_are_required(self):
        ambiguous = world(*self.objects, obj("second-anchor", "Kettle"))
        result = self.resolve("next to", {"cup-a": 1, "cup-b": 2}, snapshot=ambiguous)
        self.assertIsNone(result.target_id)
        self.assertEqual(result.reason, "anchor_not_unique")
        with self.assertRaises(ValueError):
            self.resolve("next to", {"cup-a": 1, "cup-b": 2},
                         snapshot=world(*self.objects, complete=False))


class VisibilityAndSelectionTests(unittest.TestCase):
    def test_only_nonempty_target_and_container_masks_receive_marks(self):
        snapshot = world(obj("empty"), obj("missing"), obj("visible-far", x=99),
                         obj("container", "Cabinet", openable=True, receptacle=True),
                         obj("not-container", "Book", openable=True), obj("other", "Plate"))
        marks = visible_marks(snapshot, {"empty": [], "visible-far": [(4, 4)],
                              "container": [(1, 1)], "not-container": [(2, 2)],
                              "other": [(3, 3)], "structural-id": [(0, 0)]},
                              target_type="Cup", width=5, height=5)
        self.assertEqual([(mark.label, mark.object_id) for mark in marks],
                         [(1, "container"), (2, "visible-far")])
        self.assertEqual(marks[0].roles, ("container",))
        self.assertEqual(marks[1].roles, ("candidate",))

    def test_mark_order_uses_geometry_not_object_or_mask_iteration_order(self):
        objects = (obj("a-id"), obj("z-id"))
        masks = {"a-id": [(3, 3), (4, 3)], "z-id": [(1, 0), (0, 0)]}
        first = visible_marks(world(*objects), masks, target_type="Cup", width=5, height=5)
        reordered = {key: list(reversed(value)) for key, value in reversed(list(masks.items()))}
        second = visible_marks(world(*reversed(objects)), reordered,
                               target_type="Cup", width=5, height=5)
        self.assertEqual(first, second)
        self.assertEqual(first[0].object_id, "z-id")
        for mark in first:
            self.assertIn(mark.anchor_pixel, masks[mark.object_id])

    def test_overlapping_instance_masks_are_rejected(self):
        with self.assertRaises(ValueError):
            visible_marks(world(obj("a"), obj("b")), {"a": [(0, 0)], "b": [(0, 0)]},
                          target_type="Cup", width=2, height=2)

    def test_select_requires_candidate_mask_and_measured_range(self):
        candidate = Mark(1, "target", ("candidate",), (0, 0, 0, 0), (0, 0), 1)
        self.assertEqual(selection_action("target", [candidate], measured_distance_m=1.5),
                         Action("select", 1))
        for distance in (None, 1.5001):
            self.assertIsNone(selection_action("target", [candidate], measured_distance_m=distance))
        for mark in (Mark(1, "target", ("candidate",), (0, 0, 0, 0), (0, 0), 0),
                     Mark(1, "target", ("container",), (0, 0, 0, 0), (0, 0), 1)):
            self.assertIsNone(selection_action("target", [mark], measured_distance_m=1))

    def test_duplicate_select_labels_are_rejected(self):
        marks = [Mark(1, "target", ("candidate",), (0, 0, 0, 0), (0, 0), 1),
                 Mark(1, "other", ("container",), (1, 1, 1, 1), (1, 1), 1)]
        with self.assertRaises(ValueError):
            selection_action("target", marks, measured_distance_m=1)


class PathAndWitnessTests(unittest.TestCase):
    def test_proximity_does_not_replace_verified_directed_path_evidence(self):
        nodes = (Viewpoint("a", Point3(0, 0, 0)), Viewpoint("b", Point3(1, 0, 0)),
                 Viewpoint("far", Point3(3, 0, 0)))
        evidence = {("a", "b"): PathEvidence("a", "b", "world", True, "path-ab"),
                    ("b", "a"): PathEvidence("b", "a", "world", None, "unknown"),
                    ("a", "far"): PathEvidence("a", "far", "world", True, "too-long")}
        graph = build_navigation_graph(nodes, evidence, context_key="world", opened_containers=())
        self.assertEqual(graph.neighbors("a"), ("b",))
        self.assertEqual(graph.neighbors("b"), ())
        self.assertIn(("b", "a", "path_unknown"), graph.rejected_edges)
        unchecked = build_navigation_graph(nodes, {}, context_key="world", opened_containers=())
        self.assertEqual(unchecked.edges, ())
        self.assertIn(("a", "b", "path_not_checked"), unchecked.rejected_edges)

    def test_infeasible_paths_and_wrong_context_fail_closed(self):
        nodes = (Viewpoint("a", Point3(0, 0, 0)), Viewpoint("b", Point3(1, 0, 0)))
        false = PathEvidence("a", "b", "world", False, "blocked")
        graph = build_navigation_graph(nodes, {("a", "b"): false},
                                       context_key="world", opened_containers=())
        self.assertEqual(graph.neighbors("a"), ())
        self.assertIn(("a", "b", "path_infeasible"), graph.rejected_edges)
        for evidence in (PathEvidence("a", "b", "stale-world", True, "stale"),
                         PathEvidence("b", "a", "world", True, "wrong-endpoints")):
            with self.subTest(evidence=evidence), self.assertRaises(ValueError):
                build_navigation_graph(nodes, {("a", "b"): evidence},
                                       context_key="world", opened_containers=())
        with self.assertRaises(ValueError):
            NavigationGraph("world", nodes, (false,), (), ())
        with self.assertRaises(ValueError):
            navigation_transitions(graph, ViewState("a", 0, ("cabinet",)),
                                   destination_heading=lambda state, _: state.heading_index)

    @staticmethod
    def chain_plan(goal, **budgets):
        nodes = tuple(Viewpoint(f"v{i:02d}", Point3(i, 0, 0)) for i in range(13))
        evidence = {(nodes[i].viewpoint_id, nodes[i + 1].viewpoint_id):
                    PathEvidence(nodes[i].viewpoint_id, nodes[i + 1].viewpoint_id,
                                 "world", True, f"synthetic-path-{i}") for i in range(12)}
        graph = build_navigation_graph(nodes, evidence, context_key="world", opened_containers=())
        return bounded_witness(ViewState("v00", 0),
            lambda state: navigation_transitions(graph, state,
                                                  destination_heading=lambda previous, _: previous.heading_index),
            lambda state: Action("select", 1) if state.viewpoint_id == f"v{goal:02d}" else None,
            maximum_expanded_states=budgets.get("maximum_expanded_states", 512),
            maximum_actions=budgets.get("maximum_actions", 12))

    def test_witness_action_ceiling_includes_final_select(self):
        allowed = self.chain_plan(11)
        self.assertTrue(allowed.found)
        self.assertEqual(len(allowed.actions), 12)
        self.assertEqual(allowed.actions[-1], Action("select", 1))
        self.assertTrue(allowed.requires_simulator_replay)
        self.assertEqual(allowed.reason, "candidate_requires_replay")
        too_long = self.chain_plan(12)
        self.assertFalse(too_long.found)
        self.assertEqual(too_long.actions, ())
        self.assertEqual(too_long.reason, "action_budget_exhausted")

    def test_search_budget_is_explicit_and_does_not_create_witness(self):
        limited = self.chain_plan(11, maximum_expanded_states=1)
        self.assertFalse(limited.found)
        self.assertEqual(limited.reason, "search_state_budget_exhausted")
        with self.assertRaises(ValueError):
            self.chain_plan(11, maximum_actions=13)

    def test_successor_cannot_bundle_multiple_observation_actions(self):
        invalid = (Transition(Action("turn_right"), ViewState("other", 1)),
                   Transition(Action("move", 1), ViewState("other", 0, ("cabinet",))),
                   Transition(Action("open", 1), ViewState("start", 1, ("cabinet",))),
                   Transition(Action("open", 1), ViewState("start", 0, ("a", "b"))))
        for transition in invalid:
            with self.subTest(transition=transition), self.assertRaises(ValueError):
                bounded_witness(ViewState("start", 0), lambda _: [transition], lambda _: None,
                                maximum_expanded_states=10)

    def test_one_action_cannot_have_conflicting_destinations(self):
        transitions = [Transition(Action("move", 1), ViewState("a", 0)),
                       Transition(Action("move", 1), ViewState("b", 0))]
        with self.assertRaises(ValueError):
            bounded_witness(ViewState("start", 0), lambda _: transitions, lambda _: None,
                            maximum_expanded_states=10)


if __name__ == "__main__":
    unittest.main()

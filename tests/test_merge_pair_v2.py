"""Synthetic CPU checks only; temporary fixtures are not research results."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from src.tools.merge_pair_v2 import (KINDS, balanced_selection, durations,
    clone_evidence_reader, journal_reference, load_observation, producer_arguments, proposal_identity,
    publication_eligible, reference_path, scene_number, source_eligibility,
    shard_producer, verify_clone_measurements, verify_preflight, verify_source_identity)


def source(exit_code=0, status="completed_search", cleanup=None):
    return {"_meta": {"exit_code": exit_code, "end": "synthetic-finished"},
            "metrics": {"kind": "pair_v2_shard", "status": status,
                        "attempt_journal_sha256": "synthetic-present-not-read-by-eligibility",
                        "cleanup_errors": [] if cleanup is None else cleanup},
            "config": {"seed": 0, "resolution": 255}}


def candidate(kind, index, scene="FloorPlan1", source_file="results/synthetic.json"):
    return {"kind": kind, "journal_index": index, "scene": scene, "source_file": source_file,
            "proposal_identity": kind + "_" + str(index), "target_w0": "a", "target_w1": "b"}


def proposal():
    point = lambda x: {"x": x, "y": 1.0, "z": 0.0}
    return {"scene": "FloorPlan1", "identities": {"candidate_names": ["a", "b"], "anchor_name": "x"},
            "proposal": {"w0": {"spawn": {"selected_position": point(0), "receptacle_name": "table"}},
                         "w1": {"spawn": {"selected_position": point(1), "receptacle_name": "table"}}},
            "view": {"position": point(2), "yaw": 45, "horizon": 30}}


def identity_fixture():
    device = {"logical_index": 0, "gpu_index": 2, "gpu_uuid": "GPU-synthetic",
              "gpu_name": "NVIDIA H200", "vulkan_device_index": 3, "mig_mode": "Disabled"}
    allocation = {"pbs_jobid": "synthetic.hopper", "devices": [device],
                  "cuda_visible_devices": "GPU-synthetic", "cpu_affinity": list(range(12))}
    runtime = {key: value for key, value in device.items() if key != "logical_index"}
    runtime.update(pbs_jobid="synthetic.hopper", cuda_visible_devices="GPU-synthetic", cpu_affinity=list(range(12)))
    data = source()
    data["_meta"]["name"] = "synthetic"
    data["config"].update(name="synthetic", scene="FloorPlan1", raw_evidence_dir="/synthetic/raw", runtime=runtime)
    data["config"]["shard"] = {"name": "synthetic", "scene": "FloorPlan1", "seed": 0, "resolution": 255,
        "gpu_identity": device, "logical_index": 0, "allocation": allocation, "pbs_jobid": "synthetic.hopper",
        "parent_cuda_visible_devices": "GPU-synthetic", "child_cuda_visible_devices": "GPU-synthetic",
        "node": "synthetic-node", "cpu_affinity": list(range(12)), "raw_evidence_dir": "/synthetic/raw"}
    capture = {"valid_shape": True, "action_success": True, "black": False,
               "shape": [255, 255, 3], "segmentation": {"valid": True}}
    gate = {"_meta": {"exit_code": 0, "end": "synthetic-finished"},
        "config": {"allocation": allocation, "pbs_jobid": "synthetic.hopper", "node": "synthetic-node",
                   "resolution": 255, "segmentation": True},
        "metrics": {"passed": True, "cleanup_errors": [], "errors": [], "expected_gpu_count": 1,
                    "completed_gpu_count": 1, "devices": [{"worker_error": None,
                        "observation": {"logical_index": 0, "gpu_identity": device, "node": "synthetic-node",
                                        "valid": True, "captures": [copy.deepcopy(capture), copy.deepcopy(capture)]}}]}}
    return copy.deepcopy((data, gate, runtime, allocation))


def clone_fixture(normalize_source_rotation=False):
    from src.envs.pair_v2_clone_policy import (make_clone_recipe, verify_clone_creation,
        clone_baseline_check, clone_pair_check, clone_replay_check)
    def xyz(x, y=0.5):
        return {"x": x, "y": y, "z": 0.0}
    def poses(objects):
        return [{"objectName": o["name"], "position": copy.deepcopy(o["position"]),
                 "rotation": copy.deepcopy(o["rotation"])} for o in objects if o["pickupable"]]
    evidence = {}
    def put(name, value):
        filename = "/synthetic/raw/" + name + ".json"
        evidence[filename] = copy.deepcopy(value)
        return {"file": filename, "sha256": hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()}
    def read(ref):
        return copy.deepcopy(evidence[ref["file"]])
    original = [{"name": name, "objectType": category, "objectId": name + "_id", "position": xyz(x),
                 "rotation": xyz(0, 0), "pickupable": pickup, "moveable": False, "receptacle": not pickup,
                 "parentReceptacles": ["Table_id"]}
                for name, category, x, pickup in (("A", "Mug", 0, True), ("X", "Apple", 2, True), ("Table", "Table", 1, False))]
    source = {"scene": "FloorPlan1", "objects": original, "poses": poses(original)}
    source_ref = put("original_baseline", source)
    preflight = {"snapshot": {**source, "reference": source_ref}}
    recipe = make_clone_recipe(original, "A", xyz(0.4, 0.7))
    created = copy.deepcopy(original) + [dict(copy.deepcopy(original[0]), name="native_copy", objectId="copy_id", position=xyz(0.4, 0.7))]
    if normalize_source_rotation:
        created[0]["rotation"]["x"] = 0.00003  # Native readback, not an intended second edit.
    creation_action = {"action": "SetObjectPoses", "exception": None,
        "arguments": {"objectPoses": recipe["objectPoses"], "placeStationary": False},
        "after_metadata": {"lastActionSuccess": True, "objects": created}}
    creation_ref = put("create", creation_action)
    lineage = verify_clone_creation(original, created, recipe, action_success=True)
    clone_name = lineage["clone_name"]
    sample, pivot = xyz(0.4, 0.7), xyz(0.4)
    settled = copy.deepcopy(created)
    settled[-1]["position"] = pivot
    requests = poses(settled)
    query_ref = put("spawn", {"action": "GetSpawnCoordinatesAboveReceptacle", "exception": None,
        "after_metadata": {"lastActionSuccess": True, "actionReturn": [sample]}})
    place_args = {"objectId": "copy_id", "position": sample, "rotation": xyz(0, 0), "forceKinematic": True}
    placement_ref = put("place", {"action": "PlaceObjectAtPoint", "exception": None, "arguments": place_args,
        "after_metadata": {"lastActionSuccess": True, "objects": settled}})
    proof = {"action": "GetSpawnCoordinatesAboveReceptacle", "success": True, "receptacle_name": "Table",
        "selected_position": sample, "coordinates": [sample], "requested_position": pivot,
        "query_reference": query_ref, "geometry": {"method": "native_PlaceObjectAtPoint_forceKinematic",
            "object_name": clone_name, "source_rotation": xyz(0, 0), "placement_action": "PlaceObjectAtPoint",
            "placement_arguments": place_args, "placement_success": True, "returned_position": pivot,
            "returned_rotation": xyz(0, 0), "placement_reference": placement_ref,
            "captured_before_complete_pose_set_and_settling": True}}
    final_ref = put("rest", {"action": "AdvancePhysicsStep", "exception": None,
        "after_metadata": {"lastActionSuccess": True, "isSceneAtRest": True, "objects": settled}})
    physical = {"before_objects": created, "realized_objects": settled, "requested_poses": requests,
        "edited_names": [clone_name], "placement_evidence_by_name": {clone_name: proof},
        "native_is_scene_at_rest": True, "settling": {"settled": True, "observations": [{"action_reference": final_ref}]},
        "request_reference": put("requested", {"complete_requested_poses": requests})}
    baseline = clone_baseline_check(original, created, settled, recipe, creation_action_success=True,
        requested_clone_pose=requests[-1], placement_evidence=proof, native_is_scene_at_rest=True)
    record = {"scene": "FloorPlan1", "recipe": recipe, "original_snapshot_reference": source_ref,
        "passed": True, "error": None, "world_pair_edit": False,
        "creation": {"action_reference": creation_ref, "action_success": True, "created_objects": created, "lineage": lineage},
        "construction_physical": physical, "clone_baseline_check": baseline}
    context = {"enabled": True, "setup_passed": True, "original_objects": original, "recipe": recipe,
        "lineage": lineage, "source_baseline_reference": source_ref, "creation_action_reference": creation_ref,
        "baseline_validation_reference": put("validated", record)}
    worlds, constructions, replays, restored = [], [], [], []
    for index in range(2):
        current = copy.deepcopy(settled)
        current[1]["position"] = xyz(0.8 + index * 0.4)
        worlds.append({"objects": current, "poses": poses(current), "clone_context": context})
        constructions.append({"before_objects": settled if index == 0 else worlds[0]["objects"],
                              "edited_names": ["X"], "requested_poses": poses(current)})
        recreated = put("recreate_" + str(index), creation_action)
        replay_check = clone_replay_check(original, created, recipe, lineage, action_success=True)
        replay_check["action_reference"] = recreated
        recreation_ref = put("recreation_check_" + str(index), replay_check)
        restored.append({"clone_context": context, "clone_recreation_reference": recreation_ref})
        replays.append({"clone_backend": {"opt_in": True}, "restoration": {"clone_recreation": replay_check}})
    pair_check = clone_pair_check(lineage, worlds[0]["objects"], worlds[1]["objects"],
                                 worlds[0]["poses"], constructions[1]["requested_poses"], "X")
    row = {"scene": "FloorPlan1", "identities": {"candidate_names": ["A", "native_copy"], "anchor_name": "X"},
           "clone_pair_check": pair_check}
    return (row, worlds, replays, restored, constructions, preflight, read), evidence


class MergePairV2Tests(unittest.TestCase):
    def test_clone_producer_requires_explicit_true_opt_in(self):
        command = ["python", "-u", "-m", "src.envs.pair_v2_clone_shard", "--name", "synthetic"]
        self.assertTrue(shard_producer(command, {"enable_candidate_clone": True})[1])
        for config in ({}, {"enable_candidate_clone": False}, {"enable_candidate_clone": 1}):
            with self.assertRaises(ValueError):
                shard_producer(command, config)
        with self.assertRaises(ValueError):
            shard_producer(["python", "-m", "src.envs.pair_v2_shard"], {"enable_candidate_clone": True})

    def test_clone_full_native_chain_is_independently_recomputed(self):
        args, _ = clone_fixture()
        result = verify_clone_measurements(*args)
        self.assertTrue(result["passed"])
        self.assertTrue(result["baseline"]["passed"])
        self.assertEqual(len(result["replay_recreation"]), 2)

    def test_clone_native_euler_normalization_is_not_a_new_rotation_gate(self):
        args, _ = clone_fixture(normalize_source_rotation=True)
        self.assertTrue(verify_clone_measurements(*args)["passed"])

    def test_clone_saved_pass_cannot_hide_failed_native_creation_or_falling_baseline(self):
        args, evidence = clone_fixture()
        evidence["/synthetic/raw/create.json"]["after_metadata"]["lastActionSuccess"] = False
        with self.assertRaises(ValueError):
            verify_clone_measurements(*args)
        args, evidence = clone_fixture()
        # Even a retained saved PASS cannot substitute for actual physical inputs.
        evidence["/synthetic/raw/validated.json"]["construction_physical"]["realized_objects"][0]["position"]["x"] += 0.011
        with self.assertRaises(ValueError):
            verify_clone_measurements(*args)

    def test_clone_candidate_move_is_not_an_anchor_only_world(self):
        args, _ = clone_fixture()
        args[4][0]["requested_poses"][0]["position"]["x"] += 0.001
        with self.assertRaises(ValueError):
            verify_clone_measurements(*args)

    def test_clone_replay_needs_actual_fresh_recipe_execution(self):
        args, evidence = clone_fixture()
        evidence["/synthetic/raw/recreate_1.json"]["arguments"]["placeStationary"] = True
        with self.assertRaises(ValueError):
            verify_clone_measurements(*args)
        args, _ = clone_fixture()
        args[3][1]["clone_recreation_reference"] = {"file": "/synthetic/raw/missing.json", "sha256": "missing"}
        with self.assertRaises(KeyError):
            verify_clone_measurements(*args)

    def test_original_scene_preflight_cannot_be_replaced_by_clone_inventory(self):
        args, _ = clone_fixture()
        args[5]["snapshot"]["objects"] = copy.deepcopy(args[1][0]["objects"])
        with self.assertRaises(ValueError):
            verify_clone_measurements(*args)

    def test_clone_json_index_checks_hash_and_original_mapping_without_scratch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pair_dir = root / "run/pair"
            pair_dir.mkdir(parents=True)
            archive = root / "run/clone_evidence"
            archive.mkdir()
            proof = archive / "proof.json"
            proof.write_text('{"synthetic":true}')
            digest = hashlib.sha256(proof.read_bytes()).hexdigest()
            entries = [{"file": "run/clone_evidence/proof.json", "sha256": digest,
                        "source_file": "/synthetic/raw/source.json"},
                       {"file": "run/clone_evidence/proof.json", "sha256": digest,
                        "source_file": "/synthetic/raw/identical.json"}]
            index = archive / "index.json"
            index.write_text(json.dumps({"entries": entries}))
            row = {"clone_evidence": {"schema_version": 1, "context": {"synthetic": True},
                    "index_reference": {"file": "run/clone_evidence/index.json", "sha256": hashlib.sha256(index.read_bytes()).hexdigest()}},
                   "clone_evidence_files": entries}
            reader, count = clone_evidence_reader(row, {"synthetic": True}, pair_dir, root=root, raw_root="/synthetic/raw")
            self.assertEqual(count, 2)
            self.assertEqual(reader({"file": "/synthetic/raw/source.json", "sha256": digest}), {"synthetic": True})
            with self.assertRaises(ValueError):
                reader({"file": "/synthetic/raw/source.json", "sha256": "0" * 64})
            proof.write_text('{"tampered":true}')
            with self.assertRaises(ValueError):
                clone_evidence_reader(row, {"synthetic": True}, pair_dir, root=root, raw_root="/synthetic/raw")

    def test_exact_python_module_producer_with_optional_unbuffered_flag(self):
        for flag in ([], ["-u"]):
            command = ["/synthetic/bin/python3.11", *flag, "-m", "src.envs.pair_v2_shard", "--name", "synthetic"]
            self.assertEqual(producer_arguments(command, "src.envs.pair_v2_shard"), ["--name", "synthetic"])
        for command in (["python", "-m", "other", "src.envs.pair_v2_shard"],
                        ["sh", "-m", "src.envs.pair_v2_shard"],
                        ["python", "-c", "pass", "-m", "src.envs.pair_v2_shard"]):
            with self.assertRaises(ValueError):
                producer_arguments(command, "src.envs.pair_v2_shard")

    def test_source_identity_matches_actual_runtime_and_gate(self):
        data, gate, runtime, allocation = identity_fixture()
        verify_source_identity(data, gate, logged_runtime=runtime, logged_allocation=allocation)

    def test_wrong_job_uuid_cpu_or_changed_runtime_log_is_rejected(self):
        for field, value in (("pbs_jobid", "other.hopper"), ("gpu_uuid", "GPU-other"),
                             ("cpu_affinity", list(range(1, 13)))):
            data, gate, runtime, allocation = identity_fixture()
            data["config"]["runtime"][field] = value
            with self.assertRaises(ValueError):
                verify_source_identity(data, gate, logged_runtime=runtime, logged_allocation=allocation)
        data, gate, runtime, allocation = identity_fixture()
        data["config"]["shard"]["pbs_jobid"] = "other.hopper"
        with self.assertRaises(ValueError):
            verify_source_identity(data, gate, logged_runtime=runtime, logged_allocation=allocation)

    def test_failed_before_generation_keeps_missing_attempts_unknown(self):
        data = source(1, "failed_before_generation")
        data["metrics"].pop("attempt_journal_sha256")
        data["config"]["attempt_journal"] = "data/world_pairs/v2/synthetic.jsonl"
        self.assertIsNone(journal_reference(data, source_eligibility(data)))
        self.assertNotIn("attempts", data["metrics"])
        data["metrics"]["attempts"] = 1
        with self.assertRaises(ValueError):
            journal_reference(data, source_eligibility(data))

    def test_ledger_only_failure_does_not_veto_other_complete_pairs(self):
        failed = source_eligibility(source(1, "worker_failed"))
        self.assertFalse(failed["eligible"])
        counts = {kind: 20 for kind in KINDS}
        self.assertTrue(publication_eligible(counts, [{"source": "successful"}, failed], [], []))
        self.assertFalse(publication_eligible(counts, [{"source": "successful"}, failed], ["SHA mismatch"], []))

    def test_ended_failed_source_is_explicit_partial_and_can_contribute(self):
        result = source_eligibility(source(1, "incomplete"))
        self.assertTrue(result["eligible"])
        self.assertTrue(result["partial_source"])
        self.assertEqual(result["exit_code"], 1)

    def test_running_or_unknown_cleanup_source_never_contributes(self):
        for status in ("running", "worker_failed", "failed_before_generation"):
            self.assertFalse(source_eligibility(source(1, status))["eligible"])
        value = source()
        value["metrics"].pop("cleanup_errors")
        self.assertFalse(source_eligibility(value)["eligible"])
        self.assertFalse(source_eligibility(source(cleanup=["ownership unknown"]))["eligible"])

    def test_missing_formal_finish_or_wrong_protocol_is_rejected(self):
        for field, replacement in (("exit_code", None), ("end", None)):
            value = source()
            value["_meta"][field] = replacement
            self.assertFalse(source_eligibility(value)["eligible"])
        value = source()
        value["config"]["resolution"] = 224
        self.assertFalse(source_eligibility(value)["eligible"])

    def test_first_twenty_per_class_retains_surplus_and_numeric_order(self):
        rows = [candidate(kind, index, "FloorPlan2" if index < 21 else "FloorPlan10")
                for kind in KINDS for index in range(22)]
        selected, surplus, duplicate = balanced_selection(list(reversed(rows)))
        self.assertEqual(len(selected), 80)
        self.assertEqual(len(surplus), 8)
        self.assertEqual(duplicate, [])
        for kind in KINDS:
            self.assertEqual([r["journal_index"] for r in selected if r["kind"] == kind], list(range(20)))

    def test_duplicate_proposal_is_not_counted_twice(self):
        first = candidate("look_flip", 0)
        duplicate = {**first, "source_file": "results/retry.json"}
        selected, surplus, duplicates = balanced_selection([first, duplicate])
        self.assertEqual(len(selected), 1)
        self.assertEqual(len(duplicates), 1)
        self.assertEqual(surplus, [])

    def test_conflicting_duplicate_is_not_resolved_by_best_outcome(self):
        first = candidate("look_flip", 0)
        second = {**first, "kind": "direct_flip"}
        with self.assertRaises(ValueError):
            balanced_selection([first, second])

    def test_missing_duration_is_unknown_not_zero(self):
        self.assertEqual(durations([None, float("nan")])["mean_seconds"], None)
        result = durations([None, 0.0, 2.0, float("inf")])
        self.assertEqual(result, {"observed_count": 2, "missing_count": 2,
                                  "mean_seconds": 1.0, "total_observed_seconds": 2.0})

    def test_source_paths_do_not_change_proposal_identity(self):
        a = proposal()
        b = copy.deepcopy(a)
        b["proposal"]["w0"]["spawn"]["query_reference"] = {"file": "different/run"}
        self.assertEqual(proposal_identity(a), proposal_identity(b))
        b["proposal"]["w1"]["spawn"]["selected_position"]["x"] = 3
        self.assertNotEqual(proposal_identity(a), proposal_identity(b))

    def test_invalid_scene_fails(self):
        for name in ("FloorPlan", "FloorPlan0", "../FloorPlan1", None):
            with self.assertRaises(ValueError):
                scene_number(name)
        self.assertLess(scene_number("FloorPlan2"), scene_number("FloorPlan10"))

    def test_preflight_uses_final_rest_without_extra_transient_gate(self):
        original = [{"name": "box", "moveable": True, "position": {"x": 0, "y": 0, "z": 0}}]
        final = copy.deepcopy(original)
        final[0]["position"]["x"] = 0.01
        value = {"passed": True, "snapshot": {"objects": final, "native_is_scene_at_rest": True},
            "diagnostics": {"source_snapshot": {"objects": original},
                "settling": {"settled": True, "observations": [
                    {"native_is_scene_at_rest": False, "synthetic_transient_displacement_m": 0.03},
                    {"native_is_scene_at_rest": True}]}}}
        self.assertTrue(verify_preflight(value)["passed"])
        final[0]["position"]["x"] = 0.010001
        with self.assertRaises(ValueError):
            verify_preflight(value)

    def test_artifact_hash_and_directory_are_both_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inner = root / "owned"
            inner.mkdir()
            path = inner / "evidence.json"
            path.write_text('{"synthetic":true}')
            reference = {"file": "owned/evidence.json", "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            self.assertEqual(reference_path(reference, root=root, within=inner), path)
            with self.assertRaises(ValueError):
                reference_path({**reference, "sha256": "0" * 64}, root=root, within=inner)
            with self.assertRaises(ValueError):
                reference_path(reference, root=root, within=root / "other")

    def test_portable_observation_uses_real_files_mapping(self):
        import numpy as np
        from PIL import Image
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = {"metadata": {"objects": []}, "marks": [], "mark_labels_by_name": {},
                      "input_text": "synthetic", "history": [], "files": {}}
            portable = {}
            for field in ("rgb", "marked", "masks"):
                path = root / (field + (".npz" if field == "masks" else ".png"))
                if field == "masks":
                    np.savez_compressed(path, names=np.asarray(["x"]), masks=np.ones((1, 255, 255), dtype=bool))
                else:
                    Image.fromarray(np.full((255, 255, 3), 17, dtype=np.uint8)).save(path)
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                record["files"][field] = {"file": path.name, "sha256": digest}
                portable[path] = digest
            observed = load_observation(record, portable, root, root=root)
            self.assertEqual(observed["rgb"].shape, (255, 255, 3))
            self.assertTrue(observed["masks_by_name"]["x"].all())
            del portable[root / "masks.npz"]
            with self.assertRaises(ValueError):
                load_observation(record, portable, root, root=root)


if __name__ == "__main__":
    unittest.main()

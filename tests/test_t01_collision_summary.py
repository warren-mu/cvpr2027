"""Synthetic CPU evidence fixtures; root runs these after commit via run_logged."""
import copy
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.tools import t01_collision_summary as summary
from src.tools.t01_collision_probe import compact_case, compare_state, teleport_arguments


def metadata():
    return {"objects": [{"name": "Book_fixture", "objectId": "Book|fixture", "position": {"x": 0.0, "y": 1.0, "z": 0.0},
                         "rotation": {"x": 0.125, "y": 0.0, "z": 0.0}, "isMoving": False,
                         "unknown_semantic": {"retained": True}}],
            "agent": {"position": {"x": 0.0, "y": 0.0, "z": 0.0}}, "inventoryObjects": [],
            "isSceneAtRest": True, "currentTime": 1.0, "lastAction": "Pass", "lastActionSuccess": True,
            "errorMessage": "", "actionReturn": None}


class MetadataFixtures(unittest.TestCase):
    def test_rotation_side_effect_has_stable_name_and_exact_before_after(self):
        before, after = metadata(), metadata()
        after["objects"][0]["rotation"]["x"] = 0.12500000000000003
        after.update(currentTime=2.0, lastAction="TeleportObject", lastActionSuccess=False)
        changes = summary.metadata_changes(before, after)
        self.assertFalse(changes["full_state_exact"])
        self.assertEqual(changes["objects"][0]["name"], "Book_fixture")
        self.assertTrue(changes["objects"][0]["rotation_changed"])
        self.assertFalse(changes["objects"][0]["position_changed"])
        field = changes["objects"][0]["changes"][0]
        self.assertEqual(field["path"], "/rotation/x")
        self.assertEqual(field["before"], 0.125)
        self.assertEqual(field["after"], 0.12500000000000003)
        self.assertEqual(changes["clock"][0]["path"], "/currentTime")
        self.assertEqual(changes["other_fields"], [])

    def test_unknown_added_removed_and_inventory_fields_are_not_dropped(self):
        before, after = metadata(), metadata()
        after["objects"][0]["unknown_semantic"]["added"] = None
        del after["objects"][0]["unknown_semantic"]["retained"]
        after["inventoryObjects"] = [{"objectId": "new"}]
        changes = summary.metadata_changes(before, after)
        fields = {row["path"]: row for row in changes["objects"][0]["changes"]}
        self.assertEqual(fields["/unknown_semantic/added"]["kind"], "added")
        self.assertFalse(fields["/unknown_semantic/added"]["before_present"])
        self.assertTrue(fields["/unknown_semantic/added"]["after_present"])
        self.assertEqual(fields["/unknown_semantic/retained"]["kind"], "removed")
        self.assertEqual(changes["other_fields"][0]["path"], "/inventoryObjects")

    def test_missing_required_state_stays_unknown_and_object_order_alone_is_equal(self):
        before, after = metadata(), metadata()
        after.pop("isSceneAtRest")
        changes = summary.metadata_changes(before, after)
        self.assertFalse(changes["available"])
        self.assertIsNone(changes["full_state_exact"])
        other = copy.deepcopy(before["objects"][0])
        other.update(name="Cup_fixture", objectId="Cup|fixture")
        before["objects"].append(other)
        after = copy.deepcopy(before)
        after["objects"].reverse()
        self.assertTrue(summary.metadata_changes(before, after)["full_state_exact"])


class ArchiveFixtures(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name).resolve()
        self.source = self.repo / "scratch/t01_collision_probe/t01_fixture/123.hopper-m-02"
        self.worker_root = self.source / "FloorPlan203"
        self.worker_root.mkdir(parents=True)
        self.archive = summary.EvidenceArchive(self.source, self.repo / "logs/runs/t01_summary/evidence", self.repo)
        self.refs = []

    def raw(self, name, value=None, body=None):
        data = body if body is not None else (json.dumps(value, sort_keys=True) + "\n").encode()
        path = self.worker_root / name
        path.write_bytes(data)
        ref = {"file": str(path), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
        self.refs.append(ref)
        return ref

    def rgb(self):
        import io
        from PIL import Image
        image = Image.new("RGB", (224, 224), (1, 1, 1))
        stream = io.BytesIO()
        image.save(stream, format="PNG")
        ref = self.raw("rgb_fixture.png", body=stream.getvalue())
        return {"shape": [224, 224, 3], "dtype": "uint8", "nonzero_values": 224 * 224 * 3,
                "min": 1, "max": 1, "valid": True, "black": False,
                "rgb_sha256": hashlib.sha256(image.tobytes()).hexdigest(), "image": ref}

    def case(self, *, rotation_change=True, omit_teleport=False):
        rgb = self.rgb()
        before, after = metadata(), metadata()
        after.update(currentTime=2.0, lastAction="CollidersObjectCollidingWith", actionReturn=["Col"])
        old_action = {"index": 0, "action": "CollidersObjectCollidingWith", "arguments": {"objectId": "Book|fixture"},
                      "before_metadata": before, "after_metadata": after, "before_rgb": rgb, "after_rgb": rgb,
                      "lastActionSuccess": True, "errorMessage": "", "actionReturn": ["Col"], "exception": None,
                      "state_comparison": compare_state(before, after), "rgb_bytes_exact": True}
        old_ref = self.raw("action_00000.json", old_action)
        def case_action(action, ref, old):
            return {"action_record": ref, "lastActionSuccess": action["lastActionSuccess"],
                    "errorMessage": action["errorMessage"], "actionReturn": action["actionReturn"],
                    "known_return_schema": True if old else None, "api_response_known": True,
                    "native_collision_rejected": not old, "state_comparison": action["state_comparison"],
                    "rgb_bytes_exact": True}
        new_source = None
        if not omit_teleport:
            final = copy.deepcopy(after)
            final.update(currentTime=3.0, lastAction="TeleportObject", lastActionSuccess=False,
                         errorMessage="Book|fixture is colliding with Table|fixture after teleport.", actionReturn=None)
            if rotation_change:
                final["objects"][0]["rotation"]["x"] = 0.12500000000000003
            new_action = {"index": 1, "action": "TeleportObject", "arguments": teleport_arguments(after["objects"][0]),
                "before_metadata": after, "after_metadata": final, "before_rgb": rgb, "after_rgb": rgb,
                "lastActionSuccess": False, "errorMessage": final["errorMessage"], "actionReturn": None,
                "exception": None, "state_comparison": compare_state(after, final), "rgb_bytes_exact": True}
            new_ref = self.raw("action_00001.json", new_action)
            new_source = case_action(new_action, new_ref, False)
        complete = not rotation_change and not omit_teleport
        raw = {"object_name": "Book_fixture", "object_id": "Book|fixture", "old_query": case_action(old_action, old_ref, True),
               "identity_teleport": new_source, "complete": complete, "safe_to_continue_diagnostic": complete,
               "physical_validity": None}
        if not complete:
            raw["stopped_reason"] = "synthetic incomplete/side-effect observation"
        ref = self.raw("object_0000.json", raw)
        return {**compact_case(raw), "raw_case": ref}

    def test_raw_case_action_rgb_chain_archived_with_null_physical_validity(self):
        compact = self.case()
        row = summary.describe_case(compact, self.archive, self.worker_root)
        self.assertEqual(row["outcome_pair"], "nonempty__native_collision_rejected")
        self.assertIsNone(row["physical_validity"])
        self.assertFalse(row["complete"])
        self.assertTrue(row["old_query"]["reported_comparison"]["clock_only_difference"])
        self.assertFalse(row["identity_teleport"]["reported_comparison"]["clock_only_difference"])
        self.assertTrue(row["identity_teleport"]["rgb_bytes_exact"])
        self.assertEqual(row["identity_teleport"]["metadata_changes"]["objects"][0]["name"], "Book_fixture")
        for entry in self.archive.entries.values():
            copied = (self.repo / entry["portable_file"]).read_bytes()
            self.assertEqual(hashlib.sha256(copied).hexdigest(), entry["sha256"])
            self.assertTrue(Path(entry["original_file"]).exists())
        index = self.archive.finish()
        self.assertTrue((self.repo / index["file"]).exists())

    def test_unobserved_teleport_is_not_a_success_or_collision_rejection(self):
        row = summary.describe_case(self.case(omit_teleport=True), self.archive, self.worker_root)
        self.assertIsNone(row["identity_teleport"])
        self.assertEqual(row["identity_teleport_outcome"], "not_observed")
        self.assertIsNone(row["physical_validity"])

    def test_action_hash_tamper_and_missing_evidence_fail_closed(self):
        compact = self.case()
        path = Path(compact["identity_teleport"]["action_record"]["file"])
        path.write_bytes(path.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "byte count"):
            summary.describe_case(compact, self.archive, self.worker_root)
        path.unlink()
        with self.assertRaises(FileNotFoundError):
            summary.describe_case(compact, self.archive, self.worker_root)

    def test_compact_raw_object_name_mismatch_fails_closed(self):
        compact = self.case()
        compact["object_name"] = "Wrong_fixture"
        with self.assertRaisesRegex(ValueError, "compact case"):
            summary.describe_case(compact, self.archive, self.worker_root)

    def test_path_escape_cap_and_conflicting_refs_preserve_previous_copies(self):
        ref = self.raw("first.json", {"fixture": True})
        entry = self.archive.add(ref, self.worker_root)
        copied = self.repo / entry["portable_file"]
        self.archive.total_limit = self.archive.total_bytes
        next_ref = self.raw("second.json", {"fixture": "second"})
        with self.assertRaisesRegex(ValueError, "total-size"):
            self.archive.add(next_ref, self.worker_root)
        self.assertEqual(hashlib.sha256(copied.read_bytes()).hexdigest(), ref["sha256"])
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            self.archive.add({**ref, "sha256": "0" * 64}, self.worker_root)
        outside = self.repo / "outside.json"
        outside.write_text("{}")
        with self.assertRaises(ValueError):
            self.archive.add({"file": str(outside), "bytes": 2, "sha256": hashlib.sha256(b"{}").hexdigest()})

    def test_worker_archive_preserves_all_indexed_bytes_and_partial_case_counts(self):
        compact = self.case()
        refs = list(self.refs)
        index = self.raw("evidence_index.json", {"files": refs, "raw_bytes": sum(ref["bytes"] for ref in refs)})
        incremental = self.raw("evidence_index.jsonl", body=("\n".join(json.dumps(ref) for ref in refs + [index]) + "\n").encode())
        worker = {"scene": "FloorPlan203", "logical_index": 0, "raw_directory": str(self.worker_root),
                  "objects": [compact], "objects_expected": 2, "recorded_actions": 2,
                  "evidence_index": index, "incremental_evidence_index": incremental,
                  "status": "incomplete_diagnostic", "diagnostic_completed": False, "cleanup_errors": []}
        row = summary.summarize_worker(worker, self.archive)
        self.assertTrue(row["evidence_complete"])
        self.assertFalse(row["diagnostic_completed"])
        self.assertEqual(row["observed_object_cases"], 1)
        self.assertEqual(row["unobserved_object_cases"], 1)
        self.assertIsNone(row["physical_validity"])
        self.assertEqual(row["outcome_counts"], {"nonempty__native_collision_rejected": 1})
        self.assertEqual(len(self.archive.entries), len(refs) + 2)

    def test_missing_final_index_is_explicit_unknown_without_fabricated_zero(self):
        row = summary.summarize_worker({"raw_directory": str(self.worker_root), "objects": [],
                                        "objects_expected": None, "settled": False}, self.archive)
        self.assertFalse(row["evidence_complete"])
        self.assertIsNone(row["objects_expected"])
        self.assertIsNone(row["physical_validity"])
        self.assertNotIn("outcome_counts", row)

    def test_main_output_sources_are_p8_compatible_and_final_matches_saved_metrics(self):
        # Exercise the output adapter and a real archive. Only source loading and
        # the already separately tested allocation validator are mocked here.
        compact = self.case()
        refs = list(self.refs)
        index = self.raw("evidence_index.json", {"files": refs, "raw_bytes": sum(ref["bytes"] for ref in refs)})
        worker = {"scene": "FloorPlan203", "logical_index": 0, "raw_directory": str(self.worker_root),
                  "objects": [compact], "objects_expected": 2, "recorded_actions": 2,
                  "evidence_index": index, "status": "incomplete_diagnostic", "diagnostic_completed": False,
                  "cleanup_errors": []}
        source_metrics = {"workers": [worker], "diagnostic_completed": False, "black_frame_detected": False,
                          "errors": [], "cleanup_errors": []}
        gate_metrics = {"passed": True}
        metrics_hash = lambda value: hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
        source = {"metrics": source_metrics, "config": {"pbs_jobid": "123.hopper-m-02",
                  "gate_source": {"file": "results/gate.json"}},
                  "_meta": {"exit_code": 1, "git_commit": "1" * 40, "metrics_sha256": metrics_hash(source_metrics)}}
        gate = {"metrics": gate_metrics, "_meta": {"exit_code": 0, "metrics_sha256": metrics_hash(gate_metrics)}}
        records = []
        (self.repo / "results").mkdir()
        for name, data in (("probe", source), ("gate", gate)):
            path = self.repo / "results" / (name + ".json")
            path.write_text(json.dumps(data))
            records.append((data, {"path": str(path.relative_to(self.repo)),
                                   "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}))
        # Load the real, unmodified auditor now; main receives a simple source-
        # audit stub, while the generated aggregate is checked by actual P8.
        auditor = summary.load_module("collision_summary_output_fixture_auditor",
                                      summary.ROOT / "scripts/audit_provenance.py")
        name = "t01_main_output_fixture"
        console = io.StringIO()
        with patch.object(summary, "ROOT", self.repo), \
                patch.object(summary, "load_logged", side_effect=records), \
                patch.object(summary, "load_module", return_value=SimpleNamespace(audit_file=lambda path: {"fail": [], "warn": []})), \
                patch.object(summary, "validate_source", return_value=self.source), \
                patch("sys.argv", ["t01_collision_summary", "--name", name, "--source", "results/probe.json"]), \
                redirect_stdout(console):
            self.assertEqual(summary.main(), 0)
        output = self.repo / "results" / (name + ".json")
        result = json.loads(output.read_text())
        emitted = [line.split("FINAL_METRICS_JSON:", 1)[1] for line in console.getvalue().splitlines()
                   if line.startswith("FINAL_METRICS_JSON:")]
        self.assertEqual(json.loads(emitted[-1]), result["metrics"])
        self.assertTrue((self.repo / result["metrics"]["evidence_index"]["file"]).is_file())
        for source_ref in result["sources"]:
            self.assertEqual(source_ref["file"], source_ref["path"])
            source_data = json.loads((self.repo / source_ref["file"]).read_text())
            self.assertEqual(source_ref["metrics_sha256"], metrics_hash(source_data["metrics"]))
        # Synthetic run_logged envelope exists only within this temporary fixture.
        meta = {"producer": "scripts/run_logged.py", "name": name, "command": ["synthetic-fixture"],
                "start": "fixture", "git_commit": "1" * 40, "exit_code": 0,
                "metrics_sha256": metrics_hash(result["metrics"]), "log_path": f"logs/runs/{name}.log"}
        result["_meta"] = meta
        output.write_text(json.dumps(result))
        (self.repo / meta["log_path"]).write_text(console.getvalue() + "# exit_code: 0\n")
        (self.repo / "logs/runs" / (name + ".meta.json")).write_text(json.dumps(meta))
        with patch.object(auditor, "ROOT", str(self.repo)), patch.object(auditor, "git_has_commit", return_value=True):
            self.assertEqual(auditor.audit_file(str(output))["fail"], [])
            changed = copy.deepcopy(source)
            changed["metrics"]["changed_after_summary"] = True
            (self.repo / "results/probe.json").write_text(json.dumps(changed))
            self.assertTrue(any(message.startswith("P8 source changed") for message in auditor.audit_file(str(output))["fail"]))


class ProvenanceFixtures(unittest.TestCase):
    def test_source_gate_hash_identity_and_policy_are_required_even_for_failed_probe(self):
        repo = Path("/fixture/repo")
        name, job = "t01_fixture", "123.hopper-m-02"
        allocation = {"scratch": "/fixture/scratch", "pbs_jobid": job,
                      "devices": [{"logical_index": i, "gpu_uuid": f"GPU-fixture-{i}"} for i in range(4)]}
        root = Path(allocation["scratch"]) / "t01_collision_probe" / name / job
        ref = {"path": "results/gate.json", "sha256": "a" * 64}
        gate = {"_meta": {"exit_code": 0}, "metrics": {"passed": True},
                "config": {"pbs_jobid": job, "node": "fixture", "allocation": allocation}}
        workers = [{"scene": scene, "logical_index": i, "gpu_identity": allocation["devices"][i],
                    "pbs_jobid": job, "node": "fixture", "raw_directory": str(root / scene)}
                   for i, scene in enumerate(summary.SCENES)]
        source = {"_meta": {"name": name, "exit_code": 1, "host": "fixture",
                            "command": ["python3", "-m", "src.tools.t01_collision_probe", "--name", name]},
                  "metrics": {"exit_code": 1, "kind": "t01_collision_probe", "physical_validity": None, "workers": workers},
                  "config": {"name": name, "pbs_jobid": job, "scenes": list(summary.SCENES), "allocation": allocation,
                             "raw_directory": str(root), "gate_source": {"file": ref["path"], "sha256": ref["sha256"]},
                             "state_bookkeeping_exclusions": sorted(summary.BOOKKEEPING)}}
        self.assertEqual(summary.validate_source(source, gate, ref, repo), root)
        for mutate in (lambda d: d["config"]["gate_source"].update(sha256="bad"),
                       lambda d: d["config"].update(state_bookkeeping_exclusions=[]),
                       lambda d: d["metrics"]["workers"][0].update(node="other"),
                       lambda d: d["metrics"].update(physical_validity=True)):
            changed = copy.deepcopy(source)
            mutate(changed)
            with self.assertRaises(ValueError):
                summary.validate_source(changed, gate, ref, repo)


if __name__ == "__main__":
    unittest.main()

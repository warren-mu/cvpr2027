"""Formal-cache producer CPU fixtures, never real-render certificates."""
from contextlib import contextmanager
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.envs.pair_core import ViewState
from src.envs.pair_table import RenderedView
from src.envs.paired_env import RGBFrame
from src.envs.pair_table_cache import check_cached_state
from src.tools import t02_cache_check as cache


def mask(count, offset=0):
    return bytes(offset) + b"\x01" * count + bytes(255 * 255 - count - offset)


def view(state_id, world, *, fresh=False, mismatch=False, synthetic=True):
    return RenderedView(RGBFrame(bytes([30, 40, 50]) * (255 * 255), 255, 255),
        {"X": mask(0), "A": mask(300, 3000 if mismatch else 1000), "B": mask(300, 2000)},
        {"A": 1.0, "B": 1.0, "X": 2.0}, {"position": {"x": 0, "y": 1, "z": 0}},
        "fixture-fresh" if fresh else "fixture-cached", f"{state_id}:{world}:{fresh}",
        {"pair_id": "p", "state_id": state_id, "world_id": f"w{world}", "snapshot_sha256": str(world) * 64},
        synthetic=synthetic)


def table():
    states = {key: SimpleNamespace(captures=(view(key, 0), view(key, 1)), view=ViewState("v", index))
              for index, key in enumerate(("s0", "s1"))}
    return SimpleNamespace(pair_id="p", scene="FloorPlan1", kind="look_flip", anchor_name="X",
                           candidate_names=("A", "B"), target_names=("A", "B"), source={}, states=states)


class CacheProducerTests(unittest.TestCase):
    def test_only_exact_export_module_and_python_prefix(self):
        for prefix in (["/scratch/env/bin/python3", "-m"], ["python", "-u", "-m"]):
            cache.exact_export_command([*prefix, "src.tools.t02_export", "--name", "t02_x"], "t02_x")
        for command in (
            ["bash", "-m", "src.tools.t02_export", "--name", "t02_x"],
            ["python3", "-m", "src.tools.fake", "--name", "t02_x", "src.tools.t02_export"],
            ["python3", "-u", "-u", "-m", "src.tools.t02_export", "--name", "t02_x"],
            ["python3", "-m", "src.tools.t02_export", "--name", "other"],
        ):
            with self.assertRaises(ValueError):
                cache.exact_export_command(command, "t02_x")

    def test_reference_hash_size_and_path_are_not_optional(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = cache.save_json(root / "proof.json", {"fixture": True})
            self.assertEqual(cache.read_json_ref(reference, root, expected=root / "proof.json"), {"fixture": True})
            for bad in ({**reference, "sha256": "0" * 64}, {**reference, "bytes": reference["bytes"] + 1}):
                with self.assertRaises(ValueError):
                    cache.read_json_ref(bad, root)
            with self.assertRaises(ValueError):
                cache.read_json_ref(reference, root / "subdir")
            with self.assertRaises(ValueError):
                cache.read_json_ref(reference, root, expected=root / "another.json")

    def export_fixture(self, scratch):
        name, job = "t02_export_fixture", "707001.hopper-m-02"
        directory = Path(scratch) / "t02_export" / name / job
        pair_dir = directory / "pair_0000"
        manifest = {"file": str(pair_dir / "table/table.json"), "sha256": "d" * 64}
        package = {"pair_id": "p", "scene": "FloorPlan1", "kind": "look_flip", "instruction": "fixture",
            "anchor_name": "X", "candidate_names": ["A", "B"], "target_names": ["A", "B"],
            "initial_position": {"x": 0, "y": 1, "z": 0}, "initial_heading": 0,
            "worlds": [{"reference": {"file": "w0", "sha256": "0" * 64}},
                       {"reference": {"file": "w1", "sha256": "1" * 64}}],
            "clone_context": None, "t01": {"pair_id": "p"}, "source_config": {"fixture": True},
            "reachable_positions": [], "reachable_reference": {}, "evidence_roots": [str(scratch)],
            "source_provenance": {"container": {"file": "results/pilot.json"}, "pair": {"file": "data/pair.json"}}}
        row = {"pair_id": "p", "status": "exported", "scratch_directory": str(pair_dir), "cleanup_errors": [],
            'shadow_divergent_states': 1, 'non_anchor_divergent_states': 1, 'in_clean_subset': False,
            "table_complete": True, "native_evidence_verified": True, "pair_checks_passed": True,
            "source_package": {"file": str(pair_dir / "authenticated_source.json")},
            "detail": {"file": str(pair_dir / "checks.json")}, "table_manifest": manifest}
        data = {"_meta": {"name": name, "exit_code": 0, "command": ["python3", "-u", "-m", "src.tools.t02_export", "--name", name]},
            "config": {"name": name, "version": cache.EXPORT_VERSION, "gpus": 1, "depth": 4,
                "pbs_jobid": job, "scratch_directory": str(directory), "pair_id": ["p"]},
            "metrics": {"kind": "t02_table_export", "export_checks_complete": True, "cleanup_errors": [],
                        "errors": [], "unstarted_pair_ids": [], "pairs": [row]}}
        value = SimpleNamespace(**{key: package[key] for key in ("pair_id", "scene", "kind", "instruction", "anchor_name", "candidate_names", "target_names")},
            shadow_metadata={'shadow_divergent_states': 1, 'non_anchor_divergent_states': 1, 'in_clean_subset': False},
            synthetic=False, source={"export_version": cache.EXPORT_VERSION, "t01": package["source_provenance"],
                "frozen_render_version": "w0_nonanchor_frozen_table_v1", "render_source_worlds": [world["reference"] for world in package["worlds"]]},
            initial_state="s0", states={"s0": SimpleNamespace(view=ViewState("v", 0))}, positions={"v": package["initial_position"]})
        return data, package, value, manifest

    def load_fixture(self, data, package, value, manifest, scratch, paths=None):
        def read(reference, root, *, expected=None):
            return copy.deepcopy(package) if Path(reference["file"]).name == "authenticated_source.json" else {
                "table_manifest": manifest, **value.shadow_metadata,
                'pair_checks': dict(value.shadow_metadata), 'native_checks': dict(value.shadow_metadata),
                'export_measurements': dict(value.shadow_metadata)}
        with patch.object(cache, "audited_result", side_effect=lambda path: (data, {"file": str(path)})), \
             patch("src.envs.pair_table_source.load_reviewed_pair", return_value=copy.deepcopy(package)), \
             patch.object(cache, "read_json_ref", side_effect=read), \
             patch.object(cache, "load_table", return_value=value), \
             patch.object(cache, "verify_render_sources", autospec=True, return_value={"passed": True}), \
             patch.object(cache, "check_pair", return_value={"passed": True, "synthetic_only": False}):
            return cache.load_exports(paths or ["results/export.json"], scratch)

    def test_complete_inventory_retains_export_rejections_before_cache(self):
        with tempfile.TemporaryDirectory() as scratch:
            data, package, value, manifest = self.export_fixture(scratch)
            rejected = {"pair_id": "rejected", "status": "rejected", "cleanup_errors": [],
                "scratch_directory": str(Path(data["config"]["scratch_directory"]) / "pair_0001"),
                "rejection_reasons": ["graph_mismatch"]}
            data["metrics"]["pairs"].append(rejected)
            data["config"]["pair_id"].append("rejected")
            tables, _, _, _, ledger = self.load_fixture(data, package, value, manifest, scratch)
            self.assertEqual(set(tables), {"p"})
            self.assertEqual(ledger[0]['shadow_divergent_states'], 1)
            self.assertFalse(ledger[0]['in_clean_subset'])
            self.assertEqual(ledger[1]["population_membership"], "excluded_before_cache_by_export_checks")
            self.assertEqual(ledger[1]["export_rejection_reasons"], ["graph_mismatch"])

    def test_changed_clean_subset_or_shadow_count_is_not_trusted_from_export(self):
        with tempfile.TemporaryDirectory() as scratch:
            original, package, value, manifest = self.export_fixture(scratch)
            for change in ({'in_clean_subset': True}, {'shadow_divergent_states': 0},
                           {'shadow_divergent_states': True}, {'non_anchor_divergent_states': 0}):
                data = copy.deepcopy(original)
                data['metrics']['pairs'][0].update(change)
                with self.assertRaisesRegex(ValueError, 'Shadow diagnostics'):
                    self.load_fixture(data, package, value, manifest, scratch)

    def test_unfinished_error_or_duplicate_pair_is_not_filtered_to_success(self):
        with tempfile.TemporaryDirectory() as scratch:
            original, package, value, manifest = self.export_fixture(scratch)
            for mutate in (lambda d: d["metrics"].update(unstarted_pair_ids=["pending"]),
                           lambda d: d["metrics"]["pairs"][0].update(status="error"),
                           lambda d: d["metrics"]["pairs"][0].update(cleanup_errors=["unknown"]),
                           lambda d: d["_meta"].update(exit_code=1)):
                data = copy.deepcopy(original)
                mutate(data)
                with self.assertRaises(ValueError):
                    self.load_fixture(data, package, value, manifest, scratch)
            with self.assertRaisesRegex(ValueError, "Duplicate pair"):
                self.load_fixture(original, package, value, manifest, scratch, ["results/a.json", "results/b.json"])

    def test_cache_keeps_the_export_catalog_in_its_original_order(self):
        fixed = [{"x": .25, "y": 1, "z": 0}, {"x": -.25, "y": 1, "z": 0}]
        evidence = {"reachable_positions": fixed, "reachable_reference": {"file": "query", "sha256": "a" * 64}}
        value = SimpleNamespace(states={"s": SimpleNamespace(graph_evidence=({"evidence": evidence}, {"evidence": copy.deepcopy(evidence)}))})
        self.assertEqual(cache.table_catalog(value)[0], fixed)
        value.states["s"].graph_evidence[1]["evidence"]["reachable_positions"].reverse()
        with self.assertRaises(ValueError):
            cache.table_catalog(value)

    def fresh_fixture(self, *, mismatch=False, index_error=False, clone=False):
        from src.envs import pair_v2_simulator, pair_v2_clone
        value, restores, renders = table(), [], []
        class Simulator:
            def __init__(self, *args):
                pass
            def restore_snapshot(self, snapshot):
                restores.append(snapshot)
                return None, {"restore_deterministic": True}
            def check_deadline(self):
                pass
            def finish(self):
                if index_error:
                    raise OSError("index storage failed")
                return {"file": "/fixture/index", "sha256": "a" * 64}
        class Renderer:
            def __init__(self, *args, **kwargs):
                pass
            def render_pair(self, state):
                key = "s0" if state.heading_index == 0 else "s1"
                renders.append(key)
                return tuple(view(key, world, fresh=True, mismatch=mismatch and world == 1) for world in (0, 1))
        @contextmanager
        def session(*args, **kwargs):
            yield object()
        package = {"worlds": [{"world": 0}, {"world": 1}], "clone_context": {} if clone else None,
                   "initial_position": {"x": 0, "y": 1, "z": 0}, "cached_table_manifest": {"file": "/fixture/table", "sha256": "b" * 64}}
        with tempfile.TemporaryDirectory() as directory:
            config = {"package": package, "directory": directory, "state_ids": ["s0", "s1"], "runtime": {"cpu_affinity": [0]}}
            connection = SimpleNamespace(send=lambda message: None)
            with patch.object(cache.os, "sched_setaffinity", create=True), \
                 patch.object(cache, "load_table", return_value=value), \
                 patch.object(cache, "table_catalog", return_value=([], {})), \
                 patch.object(cache, "controller_session", session), \
                 patch.object(pair_v2_simulator, "V2Simulator", side_effect=Simulator) as native, \
                 patch.object(pair_v2_clone, "V2CloneSimulator", side_effect=Simulator) as cloned, \
                 patch.object(cache, "SimulatorTableRenderer", Renderer):
                report = cache.fresh_worker(config, connection)
            self.assertEqual(native.call_count, int(not clone))
            self.assertEqual(cloned.call_count, int(clone))
            diagnostics = [json.loads(Path(row["comparison"]["file"]).read_text()) for row in report["states"]]
            self.assertTrue(all(row["cache_matches_live"] is None for row in diagnostics))
            self.assertEqual(restores, package["worlds"])
            return report, renders

    def test_new_controller_reconstructs_both_clone_worlds_before_capturing(self):
        report, renders = self.fresh_fixture(clone=True)
        self.assertEqual(report["status"], "captured")
        self.assertEqual(renders, ["s0", "s1"])

    def test_first_state_mismatch_stops_remaining_plan_without_replacement(self):
        report, renders = self.fresh_fixture(mismatch=True)
        self.assertEqual(report["status"], "cache_mismatch")
        self.assertEqual(renders, ["s0"])
        self.assertEqual(report["requested_states"], ["s0", "s1"])
        self.assertFalse(report["states"][0]["checks_passed"])

    def test_evidence_finalization_failure_is_an_error_even_after_good_captures(self):
        report, _ = self.fresh_fixture(index_error=True)
        self.assertEqual(report["status"], "error")
        self.assertIn("index storage failed", report["raw_index_error"])

    def raw_description(self, root, world):
        import numpy as np
        from PIL import Image
        frame = view("s0", world, fresh=True, synthetic=False)
        metadata = cache.save_json(root / f"meta{world}.json", {"objects": [{"name": name} for name in ("X", "A", "B")]})
        image = io.BytesIO()
        Image.frombytes("RGB", (255, 255), frame.rgb.data).save(image, format="PNG")
        png = image.getvalue()
        (root / f"rgb{world}.png").write_bytes(png)
        archive = io.BytesIO()
        np.savez_compressed(archive, names=np.array(["X", "A", "B"]),
            masks=np.stack([np.frombuffer(frame.masks[name], dtype=np.uint8).astype(bool).reshape(255, 255) for name in ("X", "A", "B")]))
        npz = archive.getvalue()
        (root / f"masks{world}.npz").write_bytes(npz)
        frame.source["capture"] = {"files": {"metadata": metadata,
            "rgb": {"file": str(root / f"rgb{world}.png"), "sha256": hashlib.sha256(png).hexdigest()},
            "masks": {"file": str(root / f"masks{world}.npz"), "sha256": hashlib.sha256(npz).hexdigest()}}}
        return cache.describe_view(frame), frame

    def test_descriptor_roundtrip_reads_actual_png_npz_and_rejects_missing_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            description, expected = self.raw_description(root, 0)
            actual = cache.decode_view(description, [root])
            self.assertEqual(actual, expected)
            with self.assertRaises(ValueError):
                cache.decode_view({**description, "synthetic": True}, [root])
            bad = copy.deepcopy(description)
            bad["mask_sha256"]["missing"] = hashlib.sha256(bytes(255 * 255)).hexdigest()
            with self.assertRaises(ValueError):
                cache.decode_view(bad, [root])
            (root / "rgb0.png").write_bytes(b"corrupt")
            with self.assertRaises(ValueError):
                cache.decode_view(description, [root])

    def test_json_roundtrip_preserves_comparison_tuples_and_exact_state_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            root, value = Path(directory), table()
            descriptions, fresh = zip(*(self.raw_description(root, world) for world in (0, 1)))
            # Pure fixture claims remain non-production: native authentication
            # is deliberately NOT mocked as passing or invoked by this test.
            value.states["s0"].captures = tuple(view("s0", world, synthetic=False) for world in (0, 1))
            comparison = check_cached_state(cache.make_sample(value, "s0", fresh))
            descriptor_ref = cache.save_json(root / "state_0000.json", {"pair_id": "p", "state_id": "s0", "worlds": descriptions})
            comparison_ref = cache.save_json(root / "comparison_0000.json", comparison)
            report = {"pair_id": "p", "requested_states": ["s0", "s1"], "states": [
                {"state_id": "s0", "capture_descriptor": descriptor_ref, "comparison": comparison_ref,
                 "checks_passed": comparison["checks_passed"], "reasons": comparison["reasons"]}]}
            samples = cache.load_samples(value, report, root, [root], ["s0", "s1"])
            self.assertEqual([sample.state_id for sample in samples], ["s0"])
            changed = copy.deepcopy(report)
            changed["states"][0]["state_id"] = "s1"
            with self.assertRaises(ValueError):
                cache.load_samples(value, changed, root, [root], ["s0", "s1"])

    def test_formal_minimum_rejected_before_any_runtime_work(self):
        argv = ["t02_cache_check", "--name", "t02_fixture", "--source", "results/export.json", "--sample-size", "199",
                "--timeout-seconds", "600", "--pair-timeout-seconds", "200"]
        with patch.object(cache.sys, "argv", argv), patch.object(cache, "execution_environment") as runtime, \
             patch.object(cache.sys, "stderr", io.StringIO()), self.assertRaises(SystemExit):
            cache.main()
        runtime.assert_not_called()


if __name__ == "__main__":
    unittest.main()

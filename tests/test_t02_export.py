"""CPU wiring fixtures only; no Controller, GPU, PBS or numerical experiment."""
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

from src.tools import t02_export as export


UUID = "GPU-11111111-2222-3333-4444-555555555555"


def package(kind="look_flip"):
    world = {"scene": "FloorPlan1", "agent_setup_command": {"position": {"x": 0, "y": 1, "z": 0}},
             "clone_context": None, "reference": {"file": "/source/world.json", "sha256": "e" * 64}}
    return {"pair_id": "p", "scene": "FloorPlan1", "kind": kind, "instruction": "Select target",
            "anchor_name": "X", "candidate_names": ["A", "B"], "target_names": ["A", "B"],
            "initial_position": {"x": 0, "y": 1, "z": 0}, "initial_heading": 0,
            "worlds": [copy.deepcopy(world), copy.deepcopy(world)], "clone_context": None,
            "reachable_positions": [{"x": .25, "y": 1, "z": 0}, {"x": -.25, "y": 1, "z": 0}],
            "reachable_reference": {"file": "/original/query.json", "sha256": "a" * 64},
            "reachable_scope": "original", "source_provenance": {"pair": {"file": "/source/pair.json",
                "sha256": "f" * 64}}, "source_config": {}, "t01": {"pair_id": "p"},
            "evidence_roots": ["/original"]}


class FakeSim:
    def __init__(self, *args):
        self.restored, self.frame_count, self.action_count = [], 0, 0
    def restore_snapshot(self, world):
        self.restored.append(world)
        return None, {"restore_deterministic": True}
    def reachable_positions(self):
        return {"positions": [{"x": 0, "y": 1, "z": .25}],
                "query_reference": {"file": "/fresh/query.json", "sha256": "b" * 64}}
    def finish(self):
        return {"file": "/fixture/raw/index.json", "sha256": "c" * 64}


class ExportTests(unittest.TestCase):
    def test_sequential_pairs_cannot_overwrite_each_others_unity_log(self):
        original = {"gpu_uuid": UUID, "cuda_visible_devices": UUID, "gpu_index": 5,
                    "log_dir": "/scratch/run/unity_logs"}
        first = export.pair_runtime(original, Path("/scratch/run/pair_0000"))
        second = export.pair_runtime(original, Path("/scratch/run/pair_0001"))
        self.assertNotEqual(Path(first["log_dir"]) / "controller/Player.log",
                            Path(second["log_dir"]) / "controller/Player.log")
        self.assertEqual(original["log_dir"], "/scratch/run/unity_logs")
        self.assertEqual(first["cuda_visible_devices"], original["cuda_visible_devices"])
        self.assertEqual(second["gpu_index"], original["gpu_index"])

    def test_requires_one_real_pbs_uuid_scratch_and_twelve_cpus(self):
        env = {"PBS_JOBID": "707001.hopper-m-02", "CUDA_VISIBLE_DEVICES": UUID, "CVPR_SCRATCH": "/scratch/fixture"}
        self.assertEqual(export.execution_environment(env, range(16))["cpu_affinity"], list(range(12)))
        for key, value in (("PBS_JOBID", ""), ("CUDA_VISIBLE_DEVICES", "0"),
                           ("CUDA_VISIBLE_DEVICES", UUID + "," + UUID), ("CVPR_SCRATCH", "relative")):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                export.execution_environment({**env, key: value}, range(12))
        with self.assertRaises(ValueError):
            export.execution_environment(env, range(11))

    def gate(self):
        identity = {"pbs_jobid": "707001.hopper-m-02", "cuda_visible_devices": UUID,
                    "scratch": "/scratch/fixture", "cpu_affinity": list(range(12))}
        device = {"logical_index": 0, "gpu_index": 5, "gpu_uuid": UUID, "vulkan_device_index": 3}
        observation = {"gpu_identity": device, "valid": True, "captures": [
            {"valid": True, "action_success": True, "segmentation": {"valid": True}} for _ in range(2)]}
        data = {"_meta": {"exit_code": 0}, "config": {"node": "hopper-1", "resolution": 255,
            "segmentation": True, "allocation": {**identity, "devices": [device], "cuda_driver_uuids": [UUID],
                                                 "ai2thor_version": "5.0.0"}},
            "metrics": {"passed": True, "cleanup_errors": [], "errors": [], "expected_gpu_count": 1,
                        "completed_gpu_count": 1, "devices": [{"observation": observation, "worker_error": None}]}}
        return data, identity

    def test_gate_uses_actual_physical_and_vulkan_mapping_without_mask_change(self):
        data, identity = self.gate()
        runtime = export.gate_runtime(data, identity, "hopper-1", Path("/scratch/logs"))
        self.assertEqual(runtime["controller_gpu_device"], 5)
        self.assertEqual(runtime["vulkan_device_index"], 3)
        self.assertEqual(runtime["cuda_visible_devices"], UUID)
        for mutate in (
            lambda d: d["config"].update(node="other"),
            lambda d: d["config"]["allocation"].update(cuda_driver_uuids=[]),
            lambda d: d["metrics"]["devices"][0].update(worker_error={"type": "crash"}),
            lambda d: d["metrics"]["devices"][0]["observation"]["captures"][1]["segmentation"].update(valid=False),
            lambda d: d["metrics"].update(cleanup_errors=["unknown"]),
        ):
            changed = copy.deepcopy(data)
            mutate(changed)
            with self.assertRaises(ValueError):
                export.gate_runtime(changed, identity, "hopper-1", Path("/scratch/logs"))

    def main_order_fixture(self, *, gate_exit=0, gate_cleanup=(), fail_pair=None, retention_failure=False, sample_rejected_raw=False):
        # Exercise the real main sequence/result writer. GPU/PBS execution and
        # byte/science authentication are boundaries mocked by this fixture.
        from src.envs import pair_table_source, thor_common

        calls, children = [], []
        cleaned = [False]
        original_retention = export.rejected_raw.apply_retention
        gate_ref = {"file": "/fixture/gate.json", "sha256": "a" * 64}
        common_ref = {"file": "/fixture/checkpoint.json", "sha256": "b" * 64}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gate, identity = self.gate()
            identity["scratch"] = str(root / "scratch")
            gate["config"]["allocation"].update(identity)
            gate["_meta"]["exit_code"] = gate_exit
            gate["metrics"]["passed"] = gate_exit == 0
            argv = ["t02_export", "--name", "t02_order", "--source", "source.json",
                    "--pair-id", "p0", "--pair-id", "p1", "--timeout-seconds", "600",
                    "--pair-timeout-seconds", "200", "--gate-timeout-seconds", "180"]

            if sample_rejected_raw or retention_failure:
                argv.append("--sample-rejected-raw")

            def run_gate(name, seconds, output, deadline, sources):
                calls.append("gate")
                self.assertEqual((name, seconds, deadline), ("t02_order_gate", 180, 1000))
                self.assertEqual(sources, [])
                self.assertTrue(output.is_dir())
                sources.append(gate_ref)
                return gate, gate_ref, list(gate_cleanup)

            def authenticate(source, pair_id, *, pair_file):
                calls.append("source:" + pair_id)
                self.assertEqual(source, Path("source.json"))
                self.assertIsNone(pair_file)
                if pair_id == fail_pair:
                    raise ValueError("source evidence SHA mismatch")
                value = package()
                value.update(pair_id=pair_id, sources=[common_ref], source_config={
                    "grid_size": .1, "visibility_distance": 10, "field_of_view": 60})
                return value

            def start_worker(target, config):
                self.assertIs(target, export.pair_worker)
                cleaned[0] = False
                calls.append("worker:" + config["package"]["pair_id"])
                children.append(config)
                return SimpleNamespace(result={"pair_id": config["package"]["pair_id"],
                    "status": "exported", "cleanup_errors": []}, error=None,
                    identity={"pid": 10}, unity_identity={"pid": 11}, receive=lambda: None)

            def stop_owned(workers):
                self.assertEqual(len(workers), 1)
                cleaned[0] = True
                return []

            def after_cleanup(row, reference, *, cleanup_confirmed):
                self.assertTrue(cleaned[0])
                self.assertTrue(cleanup_confirmed)
                if retention_failure:
                    raise OSError("retention fixture failure")
                return original_retention(row, reference, cleanup_confirmed=cleanup_confirmed)

            with patch.object(export, "ROOT", root), patch.object(export.sys, "argv", argv), \
                 patch.object(export, "execution_environment", return_value=identity), \
                 patch.object(export.os, "sched_getaffinity", return_value=set(range(12)), create=True), \
                 patch.object(export.socket, "gethostname", return_value="hopper-1"), \
                 patch.object(thor_common, "install_signals"), \
                 patch.object(export.signal, "setitimer"), \
                 patch.object(export, "deadline_after", return_value=1000), \
                 patch.object(export.time, "time", return_value=400), \
                 patch.object(export, "logged_gate", side_effect=run_gate), \
                 patch.object(pair_table_source, "load_reviewed_pair", side_effect=authenticate), \
                 patch.object(export, "Worker", side_effect=start_worker), \
                 patch.object(export, "stop_workers", side_effect=stop_owned), \
                 patch.object(export.rejected_raw, "apply_retention", side_effect=after_cleanup), \
                 patch.object(export.sys, "stdout", io.StringIO()), \
                 patch.object(export.sys, "stderr", io.StringIO()):
                code = export.main()
            saved = json.loads((root / "results/t02_order.json").read_text())
        return code, saved, calls, children, gate_ref, common_ref

    def test_gate_failure_skips_all_source_checks_and_exports(self):
        for gate_exit, cleanup, expected in ((86, (), 86), (1, (), 1), (86, ("unknown",), 1)):
            with self.subTest(gate_exit=gate_exit, cleanup=cleanup):
                code, saved, calls, children, gate_ref, _ = self.main_order_fixture(
                    gate_exit=gate_exit, gate_cleanup=cleanup)
                self.assertEqual(code, expected)
                self.assertEqual(calls, ["gate"])
                self.assertEqual(children, [])
                self.assertEqual(saved["sources"], [gate_ref])
                self.assertEqual(saved["metrics"]["pairs"], [])
                self.assertEqual(saved["metrics"]["unstarted_pair_ids"], ["p0", "p1"])
                self.assertEqual(saved["metrics"]["status"], "blocked_by_render_gate")

    def test_successful_gate_still_authenticates_all_pairs_before_first_export(self):
        code, saved, calls, children, gate_ref, common_ref = self.main_order_fixture()
        self.assertEqual(code, 0)
        self.assertEqual(calls, ["gate", "source:p0", "source:p1", "worker:p0", "worker:p1"])
        self.assertEqual(saved["sources"], [gate_ref, common_ref])
        self.assertEqual(saved["metrics"]["unstarted_pair_ids"], [])
        self.assertTrue(saved["metrics"]["export_checks_complete"])
        self.assertTrue(all(child["deadline"] == 600 for child in children))
        self.assertTrue(all(child["runtime"]["cuda_visible_devices"] == UUID for child in children))
        self.assertTrue(all(child["field_of_view"] == 60 for child in children))

    def test_retention_is_explicit_opt_in_and_runs_only_after_owned_cleanup(self):
        code, saved, _, _, _, _ = self.main_order_fixture()
        self.assertEqual(code, 0)
        self.assertNotIn("raw_retention_policy", saved["config"])
        self.assertNotIn("raw_retention_counts", saved["metrics"])
        self.assertTrue(all("raw_retention" not in row for row in saved["metrics"]["pairs"]))
        code, saved, _, _, _, _ = self.main_order_fixture(sample_rejected_raw=True)
        self.assertEqual(code, 0)
        self.assertEqual(saved["config"]["raw_retention_policy"], export.rejected_raw.POLICY)
        self.assertEqual(saved["metrics"]["raw_retention_counts"]["attempts"], 2)
        self.assertTrue(all(row["raw_retention"]["cleanup_confirmed"] for row in saved["metrics"]["pairs"]))

    def test_retention_failure_preserves_started_attempt_and_stops_future_launch(self):
        code, saved, calls, children, _, _ = self.main_order_fixture(retention_failure=True)
        self.assertEqual(code, 1)
        self.assertEqual(calls, ["gate", "source:p0", "source:p1", "worker:p0"])
        self.assertEqual(len(children), 1)
        metrics = saved["metrics"]
        self.assertEqual(len(metrics["pairs"]), 1)
        self.assertEqual(metrics["pairs"][0]["status"], "exported")
        self.assertEqual(metrics["pairs"][0]["raw_retention_error"]["type"], "OSError")
        self.assertEqual(metrics["unstarted_pair_ids"], ["p1"])
        self.assertFalse(metrics["export_checks_complete"])
        self.assertEqual(metrics["raw_retention_counts"]["attempts"], 1)

    def test_later_source_failure_prevents_even_the_first_export(self):
        code, saved, calls, children, gate_ref, common_ref = self.main_order_fixture(fail_pair="p1")
        self.assertEqual(code, 1)
        self.assertEqual(calls, ["gate", "source:p0", "source:p1"])
        self.assertEqual(children, [])
        self.assertEqual(saved["sources"], [gate_ref, common_ref])
        self.assertEqual(saved["metrics"]["pairs"], [])
        self.assertEqual(saved["metrics"]["unstarted_pair_ids"], ["p0", "p1"])
        self.assertEqual(saved["metrics"]["errors"], ["ValueError: source evidence SHA mismatch"])

    def test_uncertain_successful_gate_cleanup_stops_before_source_checks(self):
        code, saved, calls, children, gate_ref, _ = self.main_order_fixture(gate_cleanup=("unknown",))
        self.assertEqual(code, 1)
        self.assertEqual(calls, ["gate"])
        self.assertEqual(children, [])
        self.assertEqual(saved["sources"], [gate_ref])
        self.assertEqual(saved["metrics"]["unstarted_pair_ids"], ["p0", "p1"])
        self.assertEqual(saved["metrics"]["cleanup_errors"], ["unknown"])

    def export_mocks(self, package_value, *, native_pass=True):
        sim, order = FakeSim(), []
        original = SimpleNamespace(export_measurements={"fixture": True})
        loaded = SimpleNamespace(export_verified=False, cache_verified=False, synthetic=False,
            shadow_metadata={'shadow_divergent_states': 1, 'non_anchor_divergent_states': 1,
                             'in_clean_subset': False})
        manifest = {"file": "/scratch/table/table.json", "sha256": "d" * 64}
        def native(table, *, evidence_roots):
            self.assertIs(table, loaded)
            self.assertFalse(table.export_verified)
            order.append("native_reload_check")
            table.export_verified = native_pass
            return {"passed": native_pass}
        with patch.object(export, "SimulatorTableRenderer") as renderer, \
             patch.object(export, "build_table", return_value=original), \
             patch.object(export, "save_table", return_value=manifest), \
             patch.object(export, "load_table", return_value=loaded) as load, \
             patch.object(export, "verify_render_sources", side_effect=native), \
             patch.object(export, "check_pair", return_value={"passed": True, "reasons": []}) as check:
            if native_pass:
                result = export.export_checked(sim, package_value, Path("/scratch/table"), ["/scratch"], 1000)
                self.assertIs(check.call_args.args[0], loaded)
                self.assertIsNone(result["cache_matches_live"])
                self.assertFalse(result["downstream_allowed"])
                self.assertEqual(result['shadow_divergent_states'], 1)
                self.assertFalse(result['in_clean_subset'])
                load.assert_called_once_with(manifest["file"], expected_sha256=manifest["sha256"])
            else:
                with self.assertRaisesRegex(ValueError, "native_evidence_verification_failed"):
                    export.export_checked(sim, package_value, Path("/scratch/table"), ["/scratch"], 1000)
                check.assert_not_called()
            return sim, renderer.call_args, order

    def test_look_preserves_original_catalog_order_and_checks_loaded_bytes(self):
        value = package()
        sim, call, order = self.export_mocks(value)
        self.assertEqual(sim.restored, value["worlds"])
        self.assertEqual(call.args[2], value["reachable_positions"])
        self.assertEqual(call.kwargs["reachable_reference"], value["reachable_reference"])
        self.assertEqual(order, ["native_reload_check"])

    def test_direct_missing_catalog_requires_actual_query_after_restore(self):
        value = package("direct_flip")
        value.update(reachable_positions=None, reachable_reference=None)
        sim, call, _ = self.export_mocks(value)
        self.assertEqual(len(sim.restored), 3)
        self.assertEqual(call.args[2], sim.reachable_positions()["positions"])
        self.assertEqual(call.kwargs["reachable_reference"], sim.reachable_positions()["query_reference"])

    def test_missing_look_or_mixed_clone_context_does_not_invent_domain(self):
        for mutate in (lambda p: p.update(reachable_positions=None, reachable_reference=None),
                       lambda p: p["worlds"][1].update(clone_context={"unrelated": True})):
            value = package()
            mutate(value)
            with patch.object(export, "build_table") as build, self.assertRaises(ValueError):
                export.export_checked(FakeSim(), value, Path("/scratch/table"), [], 1000)
            build.assert_not_called()

    def test_failed_native_verification_never_becomes_usable_table(self):
        self.export_mocks(package(), native_pass=False)

    def test_restore_failure_is_not_a_partial_complete_table(self):
        sim = FakeSim()
        sim.restore_snapshot = lambda _: (None, {"restore_deterministic": False})
        with patch.object(export, "build_table") as build, self.assertRaisesRegex(ValueError, "source_world_restore_failed"):
            export.export_checked(sim, package(), Path("/scratch/table"), [], 1000)
        build.assert_not_called()

    def worker_failure(self, failure, *, cleanup=False, clone=False):
        # Load the real subclass definition before mocking its base binding;
        # importing these modules does not initialize a Controller/GPU.
        from src.envs import pair_v2_simulator, pair_v2_clone
        @contextmanager
        def session(*args, **kwargs):
            yield object()
            if cleanup:
                raise RuntimeError("stop failed")
        with tempfile.TemporaryDirectory() as directory:
            value = package()
            value["clone_context"] = {} if clone else None
            config = {"package": value, "runtime": {"cpu_affinity": [0]}, "directory": directory,
                      "max_table_bytes": 1000}
            with patch.object(export.os, "sched_setaffinity", create=True), \
                 patch.object(export, "controller_session", session), \
                 patch.object(pair_v2_simulator, "V2Simulator", side_effect=FakeSim) as native, \
                 patch.object(pair_v2_clone, "V2CloneSimulator", side_effect=FakeSim) as cloned, \
                 patch.object(export, "export_checked", side_effect=failure):
                report = export.pair_worker(config, None)
                self.assertEqual(cloned.call_count, int(clone))
                self.assertEqual(native.call_count, int(not clone))
                saved = json.loads(Path(report["report_reference"]["file"]).read_text())
                self.assertEqual(saved["status"], report["status"])
                self.assertFalse(report["table_complete"])
                return report

    def test_known_pair_rejection_and_unexpected_error_remain_distinct(self):
        rejected = self.worker_failure(ValueError("graph_mismatch"))
        self.assertEqual(rejected["status"], "rejected")
        self.assertEqual(rejected["rejection_reasons"], ["graph_mismatch"])
        error = self.worker_failure(ValueError("missing source metadata"), clone=True)
        self.assertEqual(error["status"], "error")
        self.assertEqual(error["rejection_reasons"], [])

    def test_cleanup_uncertainty_overrides_a_scientific_rejection(self):
        report = self.worker_failure(ValueError("graph_mismatch"), cleanup=True)
        self.assertEqual(report["status"], "error")
        self.assertTrue(report["cleanup_errors"])

    def test_black_with_finalization_or_cleanup_error_is_not_retry_exit86(self):
        from src.envs.pair_v2_simulator import BlackFrame
        black = self.worker_failure(BlackFrame({"file": "/raw/black", "sha256": "a" * 64}))
        self.assertTrue(export.black_only_failure(black))
        for mutation in ({"raw_index_error": "full disk", "status": "error"},
                         {"error": {"type": "ValueError", "message": "missing proof"}},
                         {"cleanup_errors": ["unknown"]}, {"cleanup_errors": None}):
            self.assertFalse(export.black_only_failure({**black, **mutation}))

    def test_portable_black_proof_decodes_png_and_rejects_changed_bytes(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "pair/raw"
            raw.mkdir(parents=True)
            image = io.BytesIO()
            Image.new("RGB", (255, 255)).save(image, format="PNG")
            body = image.getvalue()
            (raw / "frame.png").write_bytes(body)
            image_ref = {"file": str(raw / "frame.png"), "sha256": hashlib.sha256(body).hexdigest()}
            meta_ref = export.save_json(raw / "metadata.json", {"sceneName": "FloorPlan1_physics"})
            failure = export.save_json(raw / "failure.json", {"black": True, "valid": True,
                "rgb_shape": [255, 255, 3], "rgb_sha256": hashlib.sha256(bytes(255 * 255 * 3)).hexdigest(),
                "files": {"rgb": image_ref, "metadata": meta_ref}})
            with patch.object(export, "ROOT", root):
                copied = export.preserve_black(failure, root / "pair", {"pbs_jobid": "j", "gpu_uuid": UUID},
                                               "FloorPlan1", root / "portable")
                self.assertTrue(copied["black"])
                for reference in copied["files"]:
                    self.assertEqual(export.sha_file(root / reference["file"]), reference["sha256"])
                (raw / "frame.png").write_bytes(body + b"changed")
                with self.assertRaises(ValueError):
                    export.preserve_black(failure, root / "pair", {"pbs_jobid": "j", "gpu_uuid": UUID},
                                          "FloorPlan1", root / "must_not_exist")
                self.assertFalse((root / "must_not_exist").exists())


if __name__ == "__main__":
    unittest.main()

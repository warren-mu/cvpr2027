"""Synthetic CPU guard fixtures. Execute only after commit via run_logged."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.tools import probe_pack_retry as guard

JOB = "123.hopper-m-02"
NAME = "t01_collision_fixture"
NODE = "hopper-fixture"


def reference(name):
    return {"path": f"results/{name}.json", "sha256": hashlib.sha256(name.encode()).hexdigest()}


def black_gate(gate):
    gate["_meta"]["exit_code"] = 86
    gate["metrics"].update(exit_code=86, passed=False, black_frame_detected=True)
    first = gate["metrics"]["devices"][0]["observation"]
    first["valid"] = False
    for capture in first["captures"]:
        capture.update(valid=False, black=True, nonzero_rgb_values=0)


class ProbePackFixtures(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name).resolve()
        scratch = self.repo / "scratch"
        self.allocation = {"pbs_jobid": JOB, "scratch": str(scratch), "cpu_affinity": list(range(48)),
            "devices": [{"logical_index": i, "gpu_uuid": f"GPU-fixture-{i}", "gpu_index": i,
                         "vulkan_device_index": i} for i in range(4)]}
        self.allocation["cuda_visible_devices"] = ",".join(row["gpu_uuid"] for row in self.allocation["devices"])
        inner = ["bash", "experiments/t01_collision_pack.sh", "--name", NAME]
        command = ["python3", "src/tools/record_command.py", "--name", NAME, "--", *inner]
        self.payload = {"name": NAME, "command": command, "requested_gpu_count": 4,
                        "environment": {"CVPR_SCRATCH": str(scratch)}}
        self.fields = {"project": "CFP04-SF-125", "job_state": "F", "Exit_status": "86", "exec_host": NODE + "/0*48"}
        self.outer = {"_meta": {"name": NAME, "exit_code": 86, "command": command},
                      "config": {"pbs_job_id": JOB, "command": inner},
                      "metrics": {"command_exit_code": 86}, "error": None}
        self.gate = self.make_gate(NAME + "_gate", 224, True)
        self.gate_ref = reference(NAME + "_gate")
        self.children = {}

    def make_gate(self, name, resolution, segmentation):
        args = ["python3", "-m", "src.tools.render_gate", "--name", name, "--gpus", "4", "--resolution", str(resolution)]
        if segmentation:
            args += ["--segmentation"]
        args += ["--timeout-seconds", "180"]
        rows = []
        for identity in self.allocation["devices"]:
            capture = {"shape": [resolution, resolution, 3], "dtype": "uint8", "valid_shape": True,
                       "valid": True, "action_success": True, "black": False, "nonzero_rgb_values": 1}
            rows.append({"worker_error": None, "observation": {"node": NODE, "gpu_identity": copy.deepcopy(identity),
                        "valid": True, "captures": [copy.deepcopy(capture), copy.deepcopy(capture)]}})
        return {"_meta": {"name": name, "exit_code": 0, "command": args},
                "config": {"name": name, "pbs_jobid": JOB, "node": NODE, "resolution": resolution,
                    "segmentation": segmentation, "timeout_seconds": 180, "allocation": copy.deepcopy(self.allocation)},
                "metrics": {"passed": True, "black_frame_detected": False, "exit_code": 0, "errors": [],
                            "cleanup_errors": [], "expected_gpu_count": 4, "completed_gpu_count": 4, "devices": rows}}

    def add_probe(self, code=0):
        name = NAME + "_probe"
        raw_root = Path(self.allocation["scratch"]) / "t01_collision_probe" / name / JOB
        workers = []
        for i, scene in enumerate(guard.SCENES):
            raw = str(raw_root / scene)
            workers.append({"logical_index": i, "scene": scene, "pbs_jobid": JOB, "node": NODE,
                "gpu_identity": copy.deepcopy(self.allocation["devices"][i]), "raw_directory": raw,
                "cleanup_errors": [], "worker_error": None, "worker_returned_result": True,
                "black_frame_detected": False, "black_frame_stop": None,
                "assigned_config": {"scene": scene, "logical_index": i, "cpu_affinity": list(range(i * 12, (i + 1) * 12)),
                                    "raw_directory": raw}, "errors": ["synthetic API failure"] if code == 1 else []})
        probe = {"_meta": {"name": name, "exit_code": code, "command": ["python3", "-m", "src.tools.t01_collision_probe",
                    "--name", name, "--gate-result", f"results/{NAME}_gate.json", "--timeout-seconds", "180"]},
                 "config": {"name": name, "pbs_jobid": JOB, "scenes": list(guard.SCENES), "timeout_seconds": 180,
                    "gate_source": {"file": self.gate_ref["path"], "sha256": self.gate_ref["sha256"]},
                    "allocation": copy.deepcopy(self.allocation), "raw_directory": str(raw_root)},
                 "metrics": {"exit_code": code, "kind": "t01_collision_probe", "cleanup_errors": [], "errors": [],
                             "black_frame_detected": code == 86, "workers": workers}}
        self.children[name] = probe
        return probe

    def evidence(self, row, *, portable=False, nonzero=False):
        from PIL import Image
        directory = Path(row["raw_directory"])
        directory.mkdir(parents=True)
        image = directory / "rgb_synthetic.png"
        pixels = Image.new("RGB", (224, 224), (1, 0, 0) if nonzero else (0, 0, 0))
        pixels.save(image)
        image_bytes = image.read_bytes()
        image_hash = hashlib.sha256(image_bytes).hexdigest()
        measured = {"shape": [224, 224, 3], "dtype": "uint8", "black": True, "valid": False,
                    "min": 0, "max": 0, "nonzero_values": 0, "rgb_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
                    "image": {"file": str(image), "sha256": image_hash, "bytes": len(image_bytes)}}
        record = {key: copy.deepcopy(row[key]) for key in ("logical_index", "scene", "pbs_jobid", "gpu_identity", "node")}
        record.update(measurement=measured, stage="after:Pass", image=str(image), image_sha256=image_hash,
                      metadata={"fixture_only": True, "lastAction": "Pass"})
        raw = directory / "black_frame.json"
        raw.write_text(json.dumps(record))
        raw_bytes = raw.read_bytes()
        stop = {key: value for key, value in record.items() if key != "metadata"}
        stop.update(file=str(raw), sha256=hashlib.sha256(raw_bytes).hexdigest(), bytes=len(raw_bytes))
        if portable:
            dest = self.repo / "logs/runs" / (NAME + "_probe") / f"gpu_{row['logical_index']}" / "black_frame"
            dest.mkdir(parents=True)
            for source, key, sha in ((raw, "portable_file", stop["sha256"]), (image, "portable_image", image_hash)):
                target = dest / source.name
                target.write_bytes(source.read_bytes())
                stop[key] = str(target.relative_to(self.repo))
                stop[key + "_sha256"] = sha
                source.unlink()  # Synthetic fixture proves retained portable evidence works without scratch originals.
        row.update(black_frame_detected=True, black_frame_stop=stop, worker_returned_result=False)
        return stop

    def add_pack_black(self):
        name = "t00b_" + NAME + "_pack"
        gate = self.make_gate(name + "_gate", 255, False)
        black_gate(gate)
        self.children[name + "_gate"] = gate
        command = ["bash", "experiments/t00b_pack_smoke.sh", "--name", name, "--seconds", "120",
                   "--task-timeout", "240", "--worker-timeout", "360", "--gate-timeout", "180"]
        allocation = {"name": name, "pbs_jobid": JOB, "host": NODE,
                      "gpu_uuids": [row["gpu_uuid"] for row in self.allocation["devices"]],
                      "cuda_visible_devices": self.allocation["cuda_visible_devices"]}
        pack = {"_meta": {"name": name, "exit_code": 86, "command": command},
                "config": {"name": name, "allocation": allocation},
                "metrics": {"kind": "synthetic_packing_smoke_not_research", "gate_exit_code": 86,
                            "sources": [{"file": reference(name + "_gate")["path"],
                                         "sha256": reference(name + "_gate")["sha256"]}]}}
        self.children[name] = pack
        return pack, gate

    def check(self):
        def load(repo, path, auditor):
            name = path.stem
            if name not in self.children:
                raise FileNotFoundError("Missing audited synthetic child " + name)
            return self.children[name], reference(name)
        with patch("src.tools.phase_watch.load_logged", side_effect=load) as loader:
            value = guard.validate_probe_pack_retry(self.repo, self.payload, JOB, self.fields,
                                                    self.outer, self.gate, self.gate_ref, None)
            return value, [call.args[1].stem for call in loader.call_args_list]

    def test_initial_black_uses_actual_gate_without_loading_unstarted_children(self):
        black_gate(self.gate)
        (reason, evidence), children = self.check()
        self.assertEqual(reason, "probe_pack_initial_render_gate_black_capture")
        self.assertEqual(children, [])
        self.assertEqual(evidence["initial_gate"], self.gate_ref)

    def test_probe_black_decodes_png_and_preserves_independent_source_chain(self):
        probe = self.add_probe(86)
        self.evidence(probe["metrics"]["workers"][1])
        (reason, evidence), children = self.check()
        self.assertEqual(reason, "probe_pack_probe_black_capture")
        self.assertEqual(children, [NAME + "_probe"])
        self.assertTrue(evidence["black_captures"][0]["decoded_rgb_verified_all_zero"])
        self.assertEqual(evidence["black_captures"][0]["gpu_uuid"], self.allocation["devices"][1]["gpu_uuid"])

    def test_portable_black_evidence_remains_verifiable_without_original_files(self):
        probe = self.add_probe(86)
        self.evidence(probe["metrics"]["workers"][0], portable=True)
        (reason, evidence), _ = self.check()
        self.assertEqual(reason, "probe_pack_probe_black_capture")
        self.assertTrue(evidence["black_captures"][0]["portable_copy"])

    def test_plain_failure_is_not_a_black_retry_without_later_pack_black(self):
        self.add_probe(1)
        with self.assertRaises(FileNotFoundError):
            self.check()

    def test_clean_probe_zero_or_one_can_precede_nested_pack_black(self):
        for code in (0, 1):
            with self.subTest(probe_code=code):
                self.add_probe(code)
                self.add_pack_black()
                (reason, evidence), loaded = self.check()
                self.assertEqual(reason, "probe_pack_nested_render_gate_black_capture")
                self.assertEqual(len(loaded), 3)
                self.assertIn("pack_gate", evidence)

    def test_probe_identity_cleanup_schema_or_gate_hash_mismatch_rejected(self):
        probe = self.add_probe(86)
        self.evidence(probe["metrics"]["workers"][0])
        original = copy.deepcopy(probe)
        changes = [lambda d: d["config"].update(pbs_jobid="other"),
                   lambda d: d["config"]["gate_source"].update(sha256="changed"),
                   lambda d: d["_meta"]["command"].extend(["--extra", "1"]),
                   lambda d: d["metrics"]["workers"][0].update(node="other-node"),
                   lambda d: d["metrics"]["workers"][0]["gpu_identity"].update(gpu_uuid="GPU-outside"),
                   lambda d: d["metrics"]["workers"][0].pop("cleanup_errors"),
                   lambda d: d["metrics"].update(cleanup_errors=["unknown"]),
                   lambda d: d["metrics"].update(workers=d["metrics"]["workers"][:3]),
                   lambda d: d["metrics"]["workers"][1]["assigned_config"].update(cpu_affinity=list(range(12))),
                   lambda d: d["metrics"].update(black_frame_detected=False)]
        for mutate in changes:
            changed = copy.deepcopy(original)
            mutate(changed)
            self.children[NAME + "_probe"] = changed
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                self.check()

    def test_png_can_not_be_nonblack_even_with_consistent_file_hashes(self):
        probe = self.add_probe(86)
        self.evidence(probe["metrics"]["workers"][0], nonzero=True)
        with self.assertRaisesRegex(ValueError, "nonzero RGB"):
            self.check()

    def test_hash_changed_or_portable_escape_is_rejected(self):
        probe = self.add_probe(86)
        stop = self.evidence(probe["metrics"]["workers"][0], portable=True)
        image = self.repo / stop["portable_image"]
        image.write_bytes(image.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "hashes"):
            self.check()
        stop["portable_image"] = str(self.repo / "outside.png")
        with self.assertRaises(ValueError):
            self.check()

    def test_raw_json_hash_or_claimed_identity_mismatch_is_rejected(self):
        probe = self.add_probe(86)
        stop = self.evidence(probe["metrics"]["workers"][0])
        raw = Path(stop["file"])
        original = raw.read_bytes()
        raw.write_bytes(original + b" ")
        with self.assertRaisesRegex(ValueError, "hashes"):
            self.check()
        changed = json.loads(original)
        changed["scene"] = "FloorPlan999"
        raw.write_text(json.dumps(changed))
        # Even a self-consistent file digest cannot authenticate a different worker.
        stop.update(sha256=hashlib.sha256(raw.read_bytes()).hexdigest(), bytes=raw.stat().st_size)
        with self.assertRaisesRegex(ValueError, "metadata/identity"):
            self.check()

    def test_nonblack_exit_cross_job_and_wrong_node_outer_rejected(self):
        black_gate(self.gate)
        original = copy.deepcopy((self.fields, self.outer, self.gate))
        for mutate in (lambda: self.fields.update(Exit_status="1"),
                       lambda: self.outer["config"].update(pbs_job_id="other"),
                       lambda: self.fields.update(exec_host="wrong-node/0*48"),
                       lambda: self.gate["config"].update(segmentation=False),
                       lambda: self.outer["_meta"].update(exit_code=1)):
            self.fields, self.outer, self.gate = copy.deepcopy(original)
            mutate()
            with self.assertRaises(ValueError):
                self.check()

    def test_initial_success_gate_not_inferred_from_exit_zero_alone(self):
        self.add_probe(1)
        self.add_pack_black()
        self.gate["metrics"]["devices"][2]["observation"]["captures"][0]["nonzero_rgb_values"] = 0
        with self.assertRaises(ValueError):
            self.check()

    def test_nested_pack_nonblack_cleanup_or_identity_error_never_retries(self):
        self.add_probe(0)
        pack, gate = self.add_pack_black()
        original = copy.deepcopy((pack, gate))
        name = "t00b_" + NAME + "_pack"
        mutations = [lambda p, g: p["_meta"].update(exit_code=1),
                     lambda p, g: g["metrics"].update(cleanup_errors=["unknown"]),
                     lambda p, g: g["metrics"].update(black_frame_detected=False),
                     lambda p, g: p["config"]["allocation"].update(pbs_jobid="other"),
                     lambda p, g: p["config"]["allocation"].update(gpu_uuids=["different"] * 4),
                     lambda p, g: p["metrics"]["sources"][0].update(sha256="changed"),
                     lambda p, g: g["config"]["allocation"]["devices"][0].update(gpu_uuid="outside")]
        for mutate in mutations:
            p, g = copy.deepcopy(original)
            mutate(p, g)
            self.children[name], self.children[name + "_gate"] = p, g
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                self.check()

    def test_missing_probe_is_not_replaced_by_available_nested_failure(self):
        self.add_pack_black()
        with self.assertRaises(FileNotFoundError):
            self.check()


if __name__ == "__main__":
    unittest.main()

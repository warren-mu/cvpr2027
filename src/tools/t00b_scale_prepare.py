"""Independent 2/4/8-GPU scale overlay; the queued two-GPU source stays frozen."""
import argparse
import ast
import difflib
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess

ROOT = Path(__file__).resolve().parents[2]
VAGEN_COMMIT = "4e2976d87c0f92c97f9a820803e71908cde5fb71"
VERL_COMMIT = "3f55021e08b98315213e10d10b709c7f6bbe84f7"
HASHES = {
    "vagen/trainer/ppo/ray_trainer.py": "37d069e9e85d03b91f1df6f9b99fae27cbc7761f58db7b5f429d8f0c27d6345d",
    "vagen/trainer/main_ppo.py": "806f4be93cdf672e3cb291cc9c3988368e3cb3d6533b7eb4bf6d29fd7af7ea7d",
    "vagen/server/server.py": "d5a764202c0cac82c039930716eea3b051d24ad4ee97b4c8996b62ed5cb9d84e",
    "vagen/env/navigation/env.py": "38b271ee043460d9fd1a843ac28edadde2703ca0342d3b0ee4dd3076fc8f83f3",
    "ai2thor/fifo_server.py": "434d4eac1647c2acb32542c7c513a66fca40022fcdc1d23d119f2fd4ffa6ecc3",
    "scripts/examples/vagen_base/navigation/run.sh": "6465e1ca0ec653dad41471864b59f4d1c2568ebe1c2d121e1fefb20ea1dac3c2",
    "scripts/examples/vagen_base/navigation/env_config.yaml": "79c4007664f454d8e66ed80340e0d691541fff635244b002d448107a09ab8122",
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Expected exactly one source anchor: {old!r}")
    return text.replace(old, new)


def instrument(relative, original):
    """Pure transformation; fixtures can check it without importing the stack."""
    text = original.decode()
    if relative == "vagen/trainer/ppo/ray_trainer.py":
        text = "from t00b_runtime_hooks import phase as _t00b_phase, validation_phase as _t00b_validation\n" + text
        text = replace_once(text, "    def _validate(self):", "    @_t00b_validation\n    def _validate(self):")
        text = replace_once(text, "                print(f'global_steps: {self.global_steps}')",
                            "                _t00b_phase('train_begin', self.global_steps)\n                print(f'global_steps: {self.global_steps}')")
        text = replace_once(text, "                logger.log(data=metrics, step=self.global_steps)",
                            "                logger.log(data=metrics, step=self.global_steps)\n                _t00b_phase('train_end', self.global_steps)")
    elif relative == "vagen/trainer/main_ppo.py":
        text = "from t00b_runtime_hooks import ray_environment as _t00b_ray_environment\n" + text
        text = replace_once(text, "{'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}",
                            "{'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN', **_t00b_ray_environment()}")
    elif relative == "vagen/server/server.py":
        text = "from t00b_runtime_hooks import install_server as _t00b_install_server\n" + text
        text = replace_once(text, "        self._setup_routes()", "        self._setup_routes()\n        _t00b_install_server(self.app)")
    elif relative == "vagen/env/navigation/env.py":
        text = "from t00b_runtime_hooks import navigation_controller as _t00b_controller\n" + text
        text = replace_once(text, "self.env = ai2thor.controller.Controller(**self.thor_config)",
                            "self.env = _t00b_controller(**self.thor_config)")
    elif relative == "ai2thor/fifo_server.py":
        text = "from t00b_runtime_hooks import fifo_packet as _t00b_fifo_packet\n" + text
        text = replace_once(text, "            elif field_type is FieldType.METADATA_PATCH:\n",
                            "            elif field_type is FieldType.METADATA_PATCH:\n                _t00b_fifo_packet(self, field_type.name, body)\n")
        text = replace_once(text, "            elif field_type in self.image_fields:\n",
                            "            elif field_type in self.image_fields:\n                _t00b_fifo_packet(self, field_type.name, body, metadata)\n")
    else:
        raise ValueError(f"Unrecognized instrumentation target {relative}")
    ast.parse(text)
    return text.encode()


def overlay_tree(source, target, replacements, prefix):
    """Mirror unchanged items by symlink, writing only patched source files.

    Config/data remain reachable relative to __file__; shared originals are never
    edited or removed. An existing overlay is rejected instead of overwritten.
    """
    target.mkdir(exist_ok=False)
    for child in source.iterdir():
        if child.name == "__pycache__":
            continue
        relative = prefix + "/" + child.name
        destination = target / child.name
        if relative in replacements:
            destination.write_bytes(replacements[relative])
        elif any(path.startswith(relative + "/") for path in replacements):
            overlay_tree(child, destination, replacements, relative)
        else:
            destination.symlink_to(child.resolve(), target_is_directory=child.is_dir())


def check_baseline(path, pbs_jobid):
    data = json.loads(path.read_text())
    meta, config, metrics = data.get("_meta", {}), data.get("config", {}), data.get("metrics", {})
    if (meta.get("producer") != "scripts/run_logged.py" or meta.get("exit_code") != 0
            or not data.get("acceptance_passed") or metrics.get("completed_training_steps", 0) < 20
            or config.get("variant") != "baseline" or config.get("pbs_jobid") != pbs_jobid
            or meta.get("metrics_sha256") != digest(json.dumps(metrics, sort_keys=True).encode())):
        raise RuntimeError("vllm04 requires the logged successful baseline from this same PBS allocation")
    return {"path": str(path), "sha256": digest(path.read_bytes()), "git_commit": meta.get("git_commit")}


def make_launcher(original, variant, gpus):
    if variant not in ("baseline", "vllm04") or type(gpus) is not int or gpus not in (2, 4, 8):
        raise ValueError("Unknown scale variant or GPU count")
    steps = 21 if variant == "baseline" else 6
    changes = {"trainer.total_training_steps=300": f"trainer.total_training_steps={steps}"}
    if gpus != 4:
        changes["trainer.n_gpus_per_node=4"] = f"trainer.n_gpus_per_node={gpus}"
    if gpus == 2:
        changes["actor_rollout_ref.rollout.tensor_model_parallel_size=4"] = "actor_rollout_ref.rollout.tensor_model_parallel_size=1"
    if variant == "vllm04":
        changes["actor_rollout_ref.rollout.gpu_memory_utilization=0.1"] = "actor_rollout_ref.rollout.gpu_memory_utilization=0.4"
    text = original.decode()
    for old, new in changes.items():
        text = replace_once(text, old, new)
    return text, steps, changes


def check_gate(path, pbs_jobid, gpus, mask):
    data = json.loads(path.read_text())
    meta, metrics, config = data.get("_meta", {}), data.get("metrics", {}), data.get("config", {})
    runtime = config.get("allocation") or {}
    devices = runtime.get("devices", [])
    if (meta.get("producer") != "scripts/run_logged.py" or meta.get("exit_code") != 0
            or metrics.get("passed") is not True or config.get("resolution") != 255
            or meta.get("metrics_sha256") != digest(json.dumps(metrics, sort_keys=True).encode())
            or runtime.get("pbs_jobid") != pbs_jobid or runtime.get("cuda_visible_devices") != mask
            or [d.get("logical_index") for d in devices] != list(range(gpus))
            or len({d.get("gpu_uuid") for d in devices}) != gpus):
        raise RuntimeError("Scale training requires a logged 255px all-GPU gate from this exact allocation")
    return runtime, {"path": str(path), "sha256": digest(path.read_bytes()), "git_commit": meta.get("git_commit")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--gpus", required=True, type=int, choices=(2, 4, 8))
    parser.add_argument("--gate-result", required=True, type=Path)
    parser.add_argument("--wall-seconds", required=True, type=int)
    parser.add_argument("--server-port", required=True, type=int)
    parser.add_argument("--variant", choices=("baseline", "vllm04"), required=True)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--log-dir", required=True, type=Path)
    parser.add_argument("--baseline-result", type=Path)
    parser.add_argument("--timeout", required=True, type=float)
    args = parser.parse_args()
    if not 1024 <= args.server_port <= 65535:
        raise ValueError("Invalid private Navigation server port")
    if not os.environ.get("PBS_JOBID"):
        raise RuntimeError("T00b preparation requires a PBS allocation")
    if os.environ.get("RAY_ADDRESS"):
        raise RuntimeError("Refusing RAY_ADDRESS: this run must start its own local Ray instance")
    baseline = None
    if args.variant == "vllm04":
        if args.baseline_result is None:
            raise RuntimeError("Missing same-allocation baseline result")
        baseline = check_baseline(args.baseline_result, os.environ["PBS_JOBID"])
        baseline_config = json.loads(args.baseline_result.read_text())["config"]
        if baseline_config.get("allocation_gpu_count") != args.gpus:
            raise RuntimeError("Baseline GPU topology differs from variant")
    source = Path(os.environ.get("VAGEN_DIR", ROOT / "third_party/VAGEN")).resolve()
    head = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if head != VAGEN_COMMIT:
        raise RuntimeError("Unexpected VAGEN revision")
    verl = Path(importlib.util.find_spec("verl").origin).resolve().parent.parent
    if subprocess.check_output(["git", "-C", str(verl), "rev-parse", "HEAD"], text=True).strip() != VERL_COMMIT:
        raise RuntimeError("Unexpected verl revision")
    thor = Path(importlib.util.find_spec("ai2thor").origin).resolve().parent
    if importlib.metadata.version("ai2thor") != "5.0.0":
        raise RuntimeError("Requires installed ai2thor 5.0.0")
    allocation, gate = check_gate(args.gate_result, os.environ["PBS_JOBID"], args.gpus,
                                  os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    runtime = {**allocation, "name": args.name, "log_dir": str(args.log_dir.resolve())}
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    if not work.is_relative_to(Path(runtime["scratch"]).resolve()):
        raise RuntimeError("T00b work directory must be in approved scratch")
    # Potentially large Unity Player logs remain on scratch alongside all other
    # raw training evidence; the small manifest/patch stay in the repository.
    runtime["log_dir"] = str(work / "simulator")
    args.log_dir.mkdir(parents=True, exist_ok=True)
    runtime_path = work / "runtime.json"
    runtime_path.write_text(json.dumps(runtime, indent=2) + "\n")
    patched, provenance, diffs = {}, [], []
    originals = {}
    for relative, expected in HASHES.items():
        path = thor.parent / relative if relative.startswith("ai2thor/") else source / relative
        original = path.read_bytes()
        if digest(original) != expected:
            raise RuntimeError(f"Pinned source changed: {relative}")
        originals[relative] = original
        if relative.endswith(".py"):
            changed = instrument(relative, original)
            patched[relative] = changed
            provenance.append({"path": relative, "source": str(path), "original_sha256": expected,
                               "instrumented_sha256": digest(changed)})
            diffs.extend(difflib.unified_diff(original.decode().splitlines(True), changed.decode().splitlines(True),
                                             fromfile="original/" + relative, tofile="overlay/" + relative))
    overlay = work / "overlay"
    overlay.mkdir(exist_ok=False)
    overlay_tree(source / "vagen", overlay / "vagen", patched, "vagen")
    overlay_tree(thor, overlay / "ai2thor", patched, "ai2thor")
    hook = Path(__file__).with_name("t00b_runtime_hooks.py")
    shutil.copyfile(hook, overlay / hook.name)
    patch_path = args.log_dir / "instrumentation.patch"
    patch_path.write_text("".join(diffs))
    shutil.copyfile(hook, args.log_dir / "t00b_runtime_hooks.py")
    launcher_relative = "scripts/examples/vagen_base/navigation/run.sh"
    text, steps, changes = make_launcher(originals[launcher_relative], args.variant, args.gpus)
    runtime_changes = {'rollout_manager.base_url="http://localhost:5000"':
                       f'rollout_manager.base_url="http://127.0.0.1:{args.server_port}"'}
    for old, new in runtime_changes.items():
        text = replace_once(text, old, new)
    launcher = work / "official" / launcher_relative
    launcher.parent.mkdir(parents=True, exist_ok=True)
    launcher.write_text(text)
    launcher.with_name("env_config.yaml").write_bytes(originals["scripts/examples/vagen_base/navigation/env_config.yaml"])
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", args.server_port))
    cuda_source = Path(importlib.metadata.distribution("vllm").locate_file("vllm/platforms/cuda.py"))
    cuda_sha = digest(cuda_source.read_bytes())
    if cuda_sha != "1d27d2b5d25dbf8c511f39b7031f2d90c30fd3b5c7f052ea17a7c1138bc57c2b":
        raise RuntimeError("Expected the logged vLLM 0.8.2 UUID repair")
    config = {
        "name": args.name, "variant": args.variant, "baseline": "official_vagen_base_navigation",
        "vagen_commit": VAGEN_COMMIT, "verl_commit": VERL_COMMIT,
        "official_launcher_sha256": HASHES[launcher_relative], "generated_launcher": str(launcher),
        "generated_launcher_sha256": digest(launcher.read_bytes()), "training_config_changes": changes,
        "runtime_launcher_changes": runtime_changes, "server_port": args.server_port,
        "configured_total_training_steps": steps, "required_completed_updates": steps - 1,
        "official_total_training_steps": 300, "allocation_gpu_count": args.gpus,
        "render_gate": gate, "requested_wall_seconds": args.wall_seconds,
        "rollout_tensor_parallel_size": 1 if args.gpus == 2 else 4, "additional_critic_offload": False,
        "server_gpu_topology_change": {"navigation.devices": list(range(args.gpus))},
        "watchdog_seconds": args.timeout, "pbs_jobid": os.environ["PBS_JOBID"],
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"], "allocation": runtime,
        "runtime_json": str(runtime_path), "overlay": str(overlay), "working_directory": str(work),
        "raw_evidence_directory": str(work),
        "unity_log_directory": str(work / "simulator"),
        "evidence_archive_index": str(args.log_dir / "evidence_index.json"),
        "phase_token_file": os.environ["T00B_PHASE_TOKEN_FILE"],
        "instrumentation_sources": provenance, "instrumentation_patch": str(patch_path),
        "instrumentation_patch_sha256": digest(patch_path.read_bytes()), "hook_sha256": digest(hook.read_bytes()),
        "vllm_cuda_source_sha256": cuda_sha, "same_job_baseline": baseline,
        "installed_versions": {key: importlib.metadata.version(key) for key in
                               ("torch", "vllm", "transformers", "ai2thor", "numpy", "psutil")},
        "notes": ["Fresh RGB_IMAGE wire payloads count; METADATA_PATCH cached image reuse does not.",
                  "Explicit server phase barriers attribute frames; validation within a training step is reported separately.",
                  "All raw frame/phase events are retained, including initialization and incomplete steps.",
                  "4 GPUs retain official topology; 2/8 GPUs change topology only. Configured step count changes for baseline; vllm04 additionally changes utilization 0.1 to 0.4.",
                  "Actor/ref and critic offload retain their official launcher values."]}
    (args.log_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print("T00B_CONFIG_JSON: " + json.dumps(config, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Independently verify explicit T01v2 shard evidence and publish the fixed pilot.

Run through run_logged after commit. No simulator is invoked. Selection is the
first 20 verified unique pairs per class in numeric-scene/journal/source order;
all source failures, rejected attempts and surplus qualified pairs remain in the
ledger. An ended failed run may contribute a fully verified completed pair only
when its provenance and cleanup are known. Incomplete/unowned evidence cannot.
"""
import argparse
from collections import Counter
import copy
import hashlib
import io
import json
import math
from pathlib import Path
import re
import shutil

from src.envs.thor_common import ROOT
from src.tools.t01_partition import audited_result, canonical_hash, sha_file

KINDS = ("direct_flip", "direct_stay", "look_flip", "look_stay")
PER_CLASS = 20
REVIEWS_PER_CLASS = 10


def finite_duration(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def reference_path(reference, *, root=ROOT, within=None):
    if (not isinstance(reference, dict) or not isinstance(reference.get("file"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", str(reference.get("sha256", "")))):
        raise ValueError("Artifact requires an explicit file and SHA-256")
    path = (Path(root) / reference["file"]).resolve()
    if within is not None and not path.is_relative_to(Path(within).resolve()):
        raise ValueError("Artifact escaped its declared evidence directory")
    if not path.is_file() or sha_file(path) != reference["sha256"]:
        raise ValueError("Missing or SHA-mismatched evidence: " + reference["file"])
    return path


def read_reference(reference, *, root=ROOT, within=None):
    return json.loads(reference_path(reference, root=root, within=within).read_text())


def source_eligibility(data):
    """Ended failure != unfinished source; preserve this distinction explicitly."""
    meta, metrics, config = data.get("_meta", {}), data.get("metrics", {}), data.get("config", {})
    reasons = []
    if type(meta.get("exit_code")) is not int or not meta.get("end"):
        reasons.append("source_not_formally_finished")
    if metrics.get("kind") != "pair_v2_shard" or config.get("seed") != 0 or config.get("resolution") != 255:
        reasons.append("wrong_source_kind_seed_or_resolution")
    if metrics.get("cleanup_errors") != []:
        reasons.append("source_cleanup_unknown_or_failed")
    if metrics.get("status") in (None, "running", "failed_before_generation", "worker_failed"):
        reasons.append("source_completion_or_worker_ownership_unknown")
    partial = meta.get("exit_code") != 0 or metrics.get("status") not in ("completed_search", "completed_quota")
    return {"eligible": not reasons, "partial_source": partial, "reasons": reasons,
            "exit_code": meta.get("exit_code"), "status": metrics.get("status"),
            "error": copy.deepcopy(metrics.get("error")), "cleanup_errors": copy.deepcopy(metrics.get("cleanup_errors"))}


def load_source(path):
    data, ref = audited_result(path)  # P1-P8, plus mandatory exact FINAL/P6.
    meta, config = data["_meta"], data["config"]
    expected_name = Path(ref["file"]).stem
    if meta.get("name") != expected_name or config.get("name") != expected_name:
        raise ValueError("Source filename/meta/config name mismatch")
    if any("could not run git" in warning for warning in ref["audit"].get("warn", [])):
        raise ValueError("Source commit existence could not be established")
    command = meta.get("command")
    if (not isinstance(command, list) or "--name" not in command or "--scene" not in command
            or command[command.index("--name") + 1] != expected_name
            or command[command.index("--scene") + 1] != config.get("scene")
            or not any(value in ("src.envs.pair_v2_shard", "src/envs/pair_v2_shard.py") for value in command)):
        raise ValueError("Logged producer/name/scene does not identify a v2 shard")
    if config.get("metrics_spec_sha256") != sha_file(ROOT / "docs/METRICS_SPEC.md"):
        raise ValueError("Shard measured a different metrics specification")
    log = ROOT / meta.get("log_path", f"logs/runs/{expected_name}.log")
    side = ROOT / "logs/runs" / f"{expected_name}.meta.json"
    ref.update(log_file=str(log.relative_to(ROOT)), log_sha256=sha_file(log),
               meta_file=str(side.relative_to(ROOT)), meta_sha256=sha_file(side))
    return data, ref


def scene_number(scene):
    match = re.fullmatch(r"FloorPlan([1-9][0-9]*)", str(scene))
    if not match:
        raise ValueError("Unknown scene identifier")
    return int(match.group(1))


def proposal_identity(row):
    """Deduplicate the declared construction, excluding file paths and outcomes."""
    identities, proposal, view = row["identities"], row["proposal"], row["view"]
    points = [{"receptacle_name": proposal[key]["spawn"]["receptacle_name"],
               "selected_position": proposal[key]["spawn"]["selected_position"]} for key in ("w0", "w1")]
    return canonical_hash({"scene": row["scene"], "identities": {
        "candidate_names": sorted(identities["candidate_names"]), "anchor_name": identities["anchor_name"]},
        "points": points, "view": {key: view[key] for key in ("position", "yaw", "horizon")}})


def verify_preflight(preflight):
    """Compare the original scene with its final rest; no extra transient bound."""
    before = preflight["diagnostics"]["source_snapshot"]["objects"]
    after = preflight["snapshot"]["objects"]
    old, new = {o["name"]: o for o in before}, {o["name"]: o for o in after}
    if len(old) != len(before) or len(new) != len(after) or set(old) != set(new):
        raise ValueError("Original-scene preflight inventory changed")
    editable = {n for n, o in old.items() if o.get("moveable") or o.get("pickupable")}
    current = {n for n, o in new.items() if o.get("moveable") or o.get("pickupable")}
    if not editable or editable != current:
        raise ValueError("Preflight movable/pickupable inventory missing")
    deltas = {n: math.sqrt(sum((old[n]["position"][a] - new[n]["position"][a]) ** 2 for a in ("x", "y", "z")))
              for n in editable}
    settling = preflight["diagnostics"]["settling"]
    observations = settling.get("observations") or []
    if (preflight.get("passed") is not True or settling.get("settled") is not True
            or not observations or observations[-1].get("native_is_scene_at_rest") is not True
            or preflight["snapshot"].get("native_is_scene_at_rest") is not True
            or any(not math.isfinite(d) or d > 0.01 for d in deltas.values())):
        raise ValueError("Original scene did not settle within the approved 1cm final-position bound")
    return {"passed": True, "final_position_error_m_by_name": deltas,
            "scope": "original-to-final-rest positions; no additional transient or rotation threshold"}


def load_observation(record, portable, directory, *, root=ROOT):
    import numpy as np
    from PIL import Image
    result = {key: copy.deepcopy(record[key]) for key in
              ("metadata", "marks", "mark_labels_by_name", "input_text", "history")}
    for key in ("rgb", "marked", "masks"):
        ref = record["files"][key]
        path = reference_path(ref, root=root, within=directory)
        if portable.get(path) != ref["sha256"]:
            raise ValueError("Observation file absent from the portable manifest")
        if key != "masks":
            with Image.open(path) as image:
                if image.mode != "RGB" or image.size != (255, 255):
                    raise ValueError("Portable view is not original 255x255 RGB")
                result[key] = np.asarray(image).copy()
            if not np.any(result[key]):
                raise ValueError("Portable view is black")
        else:
            with np.load(path, allow_pickle=False) as archive:
                names = archive["names"].tolist()
                masks = archive["masks"]
                if (not isinstance(names, list) or not all(isinstance(n, str) for n in names)
                        or len(set(names)) != len(names) or masks.dtype != np.bool_
                        or masks.shape != (len(names), 255, 255)):
                    raise ValueError("Invalid portable stable-name mask archive")
                result["masks_by_name"] = {name: masks[index].copy() for index, name in enumerate(names)}
    return result


def check_pair(row, pair_path, *, root=ROOT, raw_root=None):
    """Recompute the policy from portable measured inputs, never just trust PASS."""
    import numpy as np
    from src.envs.pair_v2_policy import (classify_pair, closest_target, compare_view,
        mark_observation, physical_check, prefix_check, public_copy, public_exact,
        restore_state_check, single_edit_check, witness_check)
    directory = Path(pair_path).resolve().parent
    if row.get("status") != "passed" or row.get("kind") not in KINDS or row.get("discard_reasons") != []:
        raise ValueError("Pair was not recorded as an unqualified completed pass")
    files = row.get("portable_files")
    if not isinstance(files, list) or not files:
        raise ValueError("Missing portable evidence manifest")
    portable, by_name = {}, {}
    for ref in files:
        path = reference_path(ref, root=root, within=directory)
        if path in portable or path.name in by_name:
            raise ValueError("Duplicate portable artifact identity")
        portable[path], by_name[path.name] = ref["sha256"], ref

    def read(name):
        return read_reference(by_name[name], root=root, within=directory)

    worlds = [read(f"snapshot_w{i}.json") for i in range(2)]
    replays = [read(f"witness_w{i}.json") for i in range(2)]
    restored = [read(f"restored_snapshot_w{i}.json") for i in range(2)]
    constructions = [read(f"construction_w{i}.json") for i in range(2)]
    preflight_reference = row["scene_preflight_reference"]
    preflight_path = reference_path(preflight_reference, root=root, within=directory)
    if portable.get(preflight_path) != preflight_reference["sha256"]:
        raise ValueError("Original-scene preflight is absent from the portable manifest")
    preflight = verify_preflight(json.loads(preflight_path.read_text()))
    record = read("observations.json")
    if set(record) != {"prefix", "after"} or any(len(record[key]) != 2 for key in record):
        raise ValueError("Require two actual prefix and two post-observation captures")
    prefix = [load_observation(r, portable, directory, root=root) for r in record["prefix"]]
    after = [load_observation(r, portable, directory, root=root) for r in record["after"]]
    identities = row["identities"]
    candidates, anchor = identities["candidate_names"], identities["anchor_name"]
    instruction = f"Select the {identities['target_type']} closest to the {identities['anchor_type']}."
    if row.get("instruction") != instruction:
        raise ValueError("Instruction does not match the recorded category-only task")
    for obs in (*prefix, *after):
        rebuilt = mark_observation(obs, candidates, instruction,
            public_context="Actions: move <index>; turn_left; turn_right; select <mark>.", history=())
        if not public_exact(obs, rebuilt) or obs["mark_labels_by_name"] != rebuilt["mark_labels_by_name"]:
            raise ValueError("Portable public input/marks differ from deterministic mask-derived input")
    for index, world in enumerate(worlds):
        if world.get("scene") != row["scene"] or prefix[index]["metadata"]["objects"] != world["objects"]:
            raise ValueError("Snapshot and captured prefix objects disagree")
        if constructions[index]["realized_objects"] != world["objects"]:
            raise ValueError("Physical evidence and saved world disagree")
        original = read_reference(world["reference"], root=root, within=raw_root)
        if original != {k: v for k, v in world.items() if k != "reference"}:
            raise ValueError("Portable snapshot changed from the native raw snapshot")
        restored_original = read_reference(replays[index]["restoration"]["restored_snapshot_reference"], root=root, within=raw_root)
        if restored_original != restored[index]:
            raise ValueError("Portable restored snapshot changed from the observed raw snapshot")
    target = [closest_target(w["objects"], candidates, anchor) for w in worlds]
    if [t.get("target_name") for t in target] != [row.get("target_w0"), row.get("target_w1")]:
        raise ValueError("Recorded targets differ from independently resolved targets")
    physical = [physical_check(c["before_objects"], c["requested_poses"], c["realized_objects"],
        c["edited_names"], native_is_scene_at_rest=c["native_is_scene_at_rest"],
        placement_evidence_by_name=c["placement_evidence_by_name"]) for c in constructions]
    for c in constructions:
        if c.get("settling", {}).get("settled") is not True or not c["settling"].get("observations"):
            raise ValueError("Missing actual bounded settling observations")
        if c["settling"]["observations"][-1].get("native_is_scene_at_rest") is not True:
            raise ValueError("Final physical simulation observation was not at rest")
        last = read_reference(c["settling"]["observations"][-1]["action_reference"], root=root, within=raw_root)
        if (last.get("action") != "AdvancePhysicsStep" or last.get("exception") is not None
                or last["after_metadata"].get("lastActionSuccess") is not True
                or last["after_metadata"].get("isSceneAtRest") is not True
                or last["after_metadata"]["objects"] != c["realized_objects"]):
            raise ValueError("Recorded rest/physical positions disagree with the actual final physics action")
        request = read_reference(c["request_reference"], root=root, within=raw_root)
        if request["complete_requested_poses"] != c["requested_poses"]:
            raise ValueError("Complete requested poses changed after construction")
        for name, proof in c["placement_evidence_by_name"].items():
            query = read_reference(proof["query_reference"], root=root, within=raw_root)
            geom = proof["geometry"]
            placement = read_reference(geom["placement_reference"], root=root, within=raw_root)
            if (query.get("action") != "GetSpawnCoordinatesAboveReceptacle" or query.get("exception") is not None
                    or query["after_metadata"].get("lastActionSuccess") is not True
                    or query["after_metadata"].get("actionReturn") != proof["coordinates"]
                    or placement.get("action") != "PlaceObjectAtPoint" or placement.get("exception") is not None
                    or placement["arguments"] != geom["placement_arguments"]
                    or placement["after_metadata"].get("lastActionSuccess") is not True):
                raise ValueError("Native placement evidence is inconsistent with its authenticated action")
            returned = [o for o in placement["after_metadata"]["objects"] if o["name"] == name]
            if len(returned) != 1 or returned[0]["position"] != geom["returned_position"]:
                raise ValueError("Requested pivot is not the recorded pre-settle native placement result")
    if constructions[1]["before_objects"] != worlds[0]["objects"] or constructions[1]["edited_names"] != [anchor]:
        raise ValueError("w1 was not constructed from w0 with only its anchor edited")
    single = single_edit_check(worlds[0]["poses"], constructions[1]["requested_poses"], anchor)
    look = row["kind"].startswith("look")
    shared0 = shared1 = None
    if look:
        shared_path = (Path(root) / row["shared_prefix_file"]).resolve()
        if shared_path != reference_path(record["prefix"][0]["files"]["marked"], root=root, within=directory):
            raise ValueError("Shared public prefix is not the authenticated w0 marked prefix")
        if row.get("shared_prefix_sha256") != portable.get(shared_path):
            raise ValueError("Shared prefix SHA mismatch")
        branches = row.get("shared_public_prefix_by_world")
        if not isinstance(branches, dict) or set(branches) != {"w0", "w1"} or branches["w0"] != branches["w1"]:
            raise ValueError("Both branches must reference the same complete public prefix artifact")
        public_path = reference_path(branches["w0"], root=root, within=directory)
        if portable.get(public_path) != branches["w0"]["sha256"]:
            raise ValueError("Shared public input lacks a portable manifest hash")
        public = json.loads(public_path.read_text())
        if set(public) != {"rgb", "marked", "input_text", "marks", "history"}:
            raise ValueError("Shared public input is incomplete or contains privileged fields")
        expected_public = {key: record["prefix"][0][key] for key in ("input_text", "marks", "history")}
        expected_public.update({key: record["prefix"][0]["files"][key] for key in ("rgb", "marked")})
        if public != expected_public:
            raise ValueError("Shared public input differs from the measured w0 public input")
        # Both branches use this one authenticated public input; branch-specific
        # original captures remain separate and are checked by D1 before reuse.
        shared0, shared1 = [public_copy(prefix[0])], [public_copy(prefix[0])]
    pre = prefix_check([prefix[0]], [prefix[1]], candidates, anchor,
        same_process=(type(worlds[0].get("worker_pid")) is int
            and worlds[0]["worker_pid"] == worlds[1].get("worker_pid")
            and type(worlds[0].get("unity_pid")) is int
            and worlds[0].get("unity_pid") == worlds[1].get("unity_pid")),
        agent_unchanged=worlds[0]["agent"] == worlds[1]["agent"],
        shared_prefix_w0=shared0, shared_prefix_w1=shared1)
    evidence_views = after if look else prefix
    evidence = compare_view(*evidence_views, [*candidates, anchor])
    anchor_pixels = [int(obs["masks_by_name"].get(anchor, np.zeros((255, 255), dtype=bool)).sum()) for obs in evidence_views]
    restoration = [restore_state_check(w["objects"], r["objects"], w["agent"], r["agent"])
                   for w, r in zip(worlds, restored)]
    witnesses = []
    for index, replay in enumerate(replays):
        if (replay.get("status") != "completed" or replay.get("cleanup_errors") != []
                or replay.get("error") is not None or replay.get("construction_process_id") != worlds[index].get("worker_pid")
                or replay.get("process_id") != restored[index].get("worker_pid")):
            raise ValueError("Witness not complete, independently owned and cleanly stopped")
        if replay.get("restore_deterministic") != restoration[index]["restore_deterministic"]:
            raise ValueError("Witness restoration status disagrees with actual restored snapshot")
        checked = witness_check(replay, target[index]["target_name"])
        if replay.get("planned_actions") != row["witness_actions"][index]:
            raise ValueError("Saved witness public plan differs from pair record")
        if replay != row["witnesses"][index]["replay"]:
            raise ValueError("Portable witness differs from the pair's recorded executed witness")
        # Validate terminal measurements using the actual last replay capture,
        # without launching the simulator or guessing missing process evidence.
        terminal = replay["captures"][-1]["files"]
        meta = read_reference(terminal["metadata"], root=root, within=raw_root)
        mask_file = reference_path(terminal["masks"], root=root, within=raw_root)
        selected_name = replay["selected_name"]
        selected = [o for o in meta["objects"] if o["name"] == selected_name]
        if len(selected) != 1:
            raise ValueError("Terminal selected object absent/ambiguous in actual replay metadata")
        measured_distance = math.sqrt(sum((meta["agent"]["position"][axis] - selected[0]["position"][axis]) ** 2
                                         for axis in ("x", "y", "z")))
        with np.load(mask_file, allow_pickle=False) as archive:
            names = archive["names"].tolist()
            if (len(set(names)) != len(names) or selected_name not in names
                    or archive["masks"].dtype != np.bool_ or archive["masks"].shape != (len(names), 255, 255)):
                raise ValueError("Terminal selected object has no unique measured instance mask")
            pixels = int(archive["masks"][names.index(selected_name)].sum())
        reference_path(terminal["rgb"], root=root, within=raw_root)
        if measured_distance != replay["selected_distance_m"] or pixels != replay["selected_mask_pixels"]:
            raise ValueError("Terminal witness range/visibility differs from its raw observed capture")
        terminal_target = closest_target(meta["objects"], candidates, anchor)
        if not terminal_target["passed"] or terminal_target["target_name"] != selected_name:
            raise ValueError("Terminal replay does not select the actual unique closest target")
        witnesses.append(checked)
    common = replays[0]["executed_actions"][0] if look else None
    verdict = classify_pair(row["kind"], target_w0=target[0], target_w1=target[1], prefix=pre,
        evidence=evidence, physical_w0=physical[0], physical_w1=physical[1], single_edit=single,
        witness_w0=witnesses[0], witness_w1=witnesses[1],
        restore_deterministic=all(r["restore_deterministic"] for r in restoration),
        restore_max_abs_diff=[r.get("restore_max_abs_diff") for r in replays],
        anchor_pixels_w0=anchor_pixels[0], anchor_pixels_w1=anchor_pixels[1], common_observation_action=common)
    if verdict["acceptance_passed"] is not True or row.get("validity", {}).get("acceptance_passed") is not True:
        raise ValueError("Independent pair acceptance failed: " + repr(verdict["reasons"]))
    review = row["review_image"]
    review_path = reference_path(review, root=root, within=directory)
    if portable.get(review_path) != review["sha256"] or review_path.name != "review.png":
        raise ValueError("Review image is not uniquely authenticated portable evidence")
    return {"validity": verdict, "prefix": pre, "evidence": evidence, "physical": physical,
            "scene_preflight": preflight, "restoration": restoration,
            "witnesses": [{"passed": w["passed"], "target_name": w["target_name"]} for w in witnesses],
            "portable_file_count": len(portable), "review_image": review,
            "proposal_identity": proposal_identity(row)}


def balanced_selection(qualified):
    ordered = sorted(qualified, key=lambda item: (scene_number(item["scene"]), item["journal_index"], item["source_file"]))
    selected, surplus, duplicates = [], [], []
    counts, seen = Counter(), {}
    for item in ordered:
        identity = item["proposal_identity"]
        if identity in seen:
            old = seen[identity]
            if (old["kind"], old["target_w0"], old["target_w1"]) != (item["kind"], item["target_w0"], item["target_w1"]):
                raise ValueError("Duplicate construction has conflicting class/targets")
            duplicates.append(item)
            continue
        seen[identity] = item
        if counts[item["kind"]] < PER_CLASS:
            selected.append(item)
            counts[item["kind"]] += 1
        else:
            surplus.append(item)
    return selected, surplus, duplicates


def durations(values):
    measured = [value for value in values if finite_duration(value)]
    return {"observed_count": len(measured), "missing_count": len(values) - len(measured),
            "mean_seconds": sum(measured) / len(measured) if measured else None,
            "total_observed_seconds": sum(measured) if measured else None}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="pilot_pairs_v2", choices=("pilot_pairs_v2",))
    parser.add_argument("--source", nargs="+", action="extend", required=True)
    args = parser.parse_args(argv)
    output = ROOT / "data/world_pairs/pilot_pairs_v2.jsonl"
    review_dir = ROOT / "logs/review/pilot_v2"
    result_path = ROOT / "results" / f"{args.name}.json"
    if any(path.exists() for path in (output, review_dir, result_path)):
        raise FileExistsError("Existing pilot/result/review evidence is preserved; no overwrite")
    metrics = {"kind": "pair_v2_merged_pilot", "status": "failed", "acceptance_passed": False,
        "passed_by_class": {k: 0 for k in KINDS}, "source_runs": [], "rejection_counts": {},
        "source_errors": [], "pair_errors": [], "scenes_excluded": [], "review_files": []}
    config = {"name": args.name, "requested_sources": args.source, "selection_per_class": PER_CLASS,
        "review_per_class": REVIEWS_PER_CLASS, "order": "numeric scene, journal row, source filename",
        "partial_source_policy": "formally ended failed source may supply independently complete pairs only with known clean cleanup and verified provenance",
        "metrics_spec_sha256": sha_file(ROOT / "docs/METRICS_SPEC.md"),
        "dataset": str(output.relative_to(ROOT)), "review_directory": str(review_dir.relative_to(ROOT))}
    sources, qualified, attempts, reasons = [], [], [], Counter()
    code = 1
    try:
        paths = [(ROOT / path).resolve() for path in args.source]
        if len(paths) != len(set(paths)):
            raise ValueError("Duplicate explicit source")
        for path in paths:
            try:
                data, reference = load_source(path)
                sources.append(reference)
                measured, cfg = data["metrics"], data["config"]
                eligible = source_eligibility(data)
                source = {"source": reference, **eligible, "scene": cfg["scene"],
                          "pbs_jobid": cfg.get("runtime", {}).get("pbs_jobid"), "recorded_attempts": measured.get("attempts")}
                metrics["source_runs"].append(source)
                scene_number(cfg["scene"])
                journal_ref = {"file": measured["attempt_journal"], "sha256": measured["attempt_journal_sha256"]}
                journal = reference_path(journal_ref, within=ROOT / "data/world_pairs/v2")
                if measured["attempt_journal"] != cfg["attempt_journal"]:
                    raise ValueError("Source config/metrics journal disagreement")
                source["attempt_journal"] = journal_ref
                passed_refs = measured["passed_pair_files"]
                if not isinstance(passed_refs, list):
                    raise ValueError("Missing recorded passed-pair references")
                by_path = {}
                for item in passed_refs:
                    verified = reference_path(item, within=ROOT / "data/world_pairs/v2" / cfg["name"])
                    if verified in by_path:
                        raise ValueError("Duplicate recorded passed-pair path")
                    by_path[verified] = item
                seen, matched, rows = set(), set(), []
                for line in journal.read_text().splitlines():
                    if not line.strip():
                        continue
                    compact = json.loads(line)
                    if compact.get("scene") != cfg["scene"] or compact.get("pair_id") in seen:
                        raise ValueError("Wrong-scene or duplicated attempt identity")
                    seen.add(compact["pair_id"])
                    raw = read_reference(compact["raw_attempt"], within=cfg["raw_evidence_dir"])
                    if any(raw.get(key) != compact.get(key) for key in
                           ("pair_id", "scene", "kind", "status", "discard_reasons", "generation_seconds", "pair_file", "pair_sha256")):
                        raise ValueError("Journal and authenticated raw attempt disagree")
                    attempts.append({"source_file": reference["file"], "journal_index": len(rows), **compact})
                    rows.append(compact)
                    reasons.update(compact.get("discard_reasons") or [])
                    if compact.get("status") != "passed":
                        continue
                    pair_path = (ROOT / compact["pair_file"]).resolve()
                    if pair_path not in by_path:
                        raise ValueError("Passed journal row lacks a logged pair-file SHA")
                    if compact.get("pair_sha256") != by_path[pair_path]["sha256"]:
                        raise ValueError("Passed journal/pair hash disagree")
                    matched.add(pair_path)
                    if not eligible["eligible"]:
                        reasons["source_ineligible_completed_pair_not_admitted"] += 1
                        continue
                    pair = read_reference(by_path[pair_path], within=pair_path.parent)
                    source_preflight = measured["scene_preflight_reference"]
                    reference_path(source_preflight, within=cfg["raw_evidence_dir"])
                    if pair["scene_preflight_reference"]["sha256"] != source_preflight["sha256"]:
                        raise ValueError("Pair's portable preflight differs from the source's logged preflight")
                    if any(pair.get(key) != compact.get(key) for key in ("pair_id", "scene", "kind", "status", "discard_reasons")):
                        raise ValueError("Portable pair disagrees with journal identity/outcome")
                    try:
                        checked = check_pair(pair, pair_path, raw_root=cfg["raw_evidence_dir"])
                        qualified.append({"pair_id": pair["pair_id"], "scene": pair["scene"], "kind": pair["kind"],
                            "target_w0": pair["target_w0"], "target_w1": pair["target_w1"],
                            "source_file": reference["file"], "source_metrics_sha256": reference["metrics_sha256"],
                            "partial_source": eligible["partial_source"], "journal_index": len(rows) - 1,
                            "generation_seconds": compact.get("generation_seconds"), "pair_reference": by_path[pair_path],
                            "review_image": checked["review_image"], "proposal_identity": checked["proposal_identity"],
                            "independent_verification": checked, "pair": pair})
                    except Exception as error:
                        metrics["pair_errors"].append({"source": reference["file"], "pair_id": compact["pair_id"], "error": str(error)})
                        reasons["independent_pair_verification_failed"] += 1
                if len(rows) != measured.get("attempts") or matched != set(by_path):
                    raise ValueError("Journal count or passed-pair references disagree with logged metrics")
                counted = {kind: sum(r.get("kind") == kind and r.get("status") == "passed" for r in rows) for kind in KINDS}
                if counted != measured.get("passed_by_class"):
                    raise ValueError("Logged per-class counts disagree with the journal")
                if measured.get("scene_excluded") is not None:
                    metrics["scenes_excluded"].append({"source": reference["file"], **measured["scene_excluded"]})
            except Exception as error:
                metrics["source_errors"].append({"file": str(path), "error": f"{type(error).__name__}: {error}"})
        selected, surplus, duplicates = balanced_selection(qualified)
        counts = {kind: sum(item["kind"] == kind for item in selected) for kind in KINDS}
        ratio = counts["look_flip"] / counts["look_stay"] if counts["look_stay"] else None
        eligible_complete = all(row["eligible"] for row in metrics["source_runs"])
        acceptance = (bool(sources) and eligible_complete and not metrics["source_errors"] and not metrics["pair_errors"]
                      and all(counts[k] >= PER_CLASS for k in KINDS) and ratio is not None and 0.8 <= ratio <= 1.25)
        def compact(item):
            return {k: v for k, v in item.items() if k not in ("pair", "independent_verification")}
        metrics.update(passed_by_class=counts, qualified_by_class={k: sum(r["kind"] == k for r in qualified) for k in KINDS},
            selected_pairs=[compact(r) for r in selected], qualified_not_selected=[compact(r) for r in surplus],
            duplicate_completed_pairs=[compact(r) for r in duplicates], look_flip_stay_ratio=ratio,
            attempts=len(attempts), rejection_counts=dict(reasons), scene_exclusion_count=len(metrics["scenes_excluded"]),
            attempt_generation_time=durations([r.get("generation_seconds") for r in attempts]),
            accepted_generation_time=durations([r.get("generation_seconds") for r in selected]),
            timing_scope="recorded construction durations only; no PBS wait/allocation or unmeasured setup attributed",
            source_failure_count=sum(row["partial_source"] for row in metrics["source_runs"]))
        if acceptance:
            output.parent.mkdir(parents=True, exist_ok=True)
            review_dir.mkdir(parents=True, exist_ok=False)
            with output.open("x") as stream:
                for item in selected:
                    stream.write(json.dumps({**item["pair"], "merge_source": compact(item),
                        "independent_verification": item["independent_verification"]}, sort_keys=True, allow_nan=False) + "\n")
            for kind in KINDS:
                rows = [r for r in selected if r["kind"] == kind][:REVIEWS_PER_CLASS]
                for index, row in enumerate(rows):
                    source = reference_path(row["review_image"], within=ROOT / "data/world_pairs/v2")
                    destination = review_dir / f"{kind}_{index:02d}_{row['pair_id']}.png"
                    with source.open("rb") as incoming, destination.open("xb") as outgoing:
                        shutil.copyfileobj(incoming, outgoing)
                    if sha_file(destination) != row["review_image"]["sha256"]:
                        raise ValueError("Review copy hash mismatch")
                    metrics["review_files"].append({"file": str(destination.relative_to(ROOT)), "sha256": sha_file(destination),
                                                   "pair_id": row["pair_id"], "kind": kind})
            metrics["dataset"] = {"file": str(output.relative_to(ROOT)), "sha256": sha_file(output)}
            metrics["acceptance_passed"] = len(selected) >= 80 and len(metrics["review_files"]) == 40
            metrics["status"] = "pilot_ready_for_review" if metrics["acceptance_passed"] else "output_incomplete"
            code = 0 if metrics["acceptance_passed"] else 1
        else:
            metrics["status"] = "insufficient_or_invalid_evidence"
    except Exception as error:
        metrics["source_errors"].append({"stage": "merge", "error": f"{type(error).__name__}: {error}"})
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open("x") as stream:
        json.dump({"metrics": metrics, "config": config, "sources": sources}, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print("FINAL_METRICS_JSON: " + json.dumps(metrics, sort_keys=True, allow_nan=False), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())

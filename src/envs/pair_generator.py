"""Bounded T01 pilot generation with explicit strict/pending/rejected outcomes.

Use ``python -m src.envs.pair_generator`` through run_logged, after committing
code. No simulator imports or measurements run on import. Every attempted edit
is journaled; nonzero exit preserves partial evidence. Required scientific
thresholds are supplied explicitly, never inferred from the observed output.
"""
import argparse
from collections import Counter
from dataclasses import asdict
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import time
import traceback

from .pair_core import (RelationalTask, ViewState, make_scene_split,
                        navigation_transitions)
from .pair_simulator import (BUILD, PairSimulator, compare_inputs, digest,
                             point, pose_changes, save_observation)
from .thor_common import (ROOT, Worker, controller_session, deadline_after,
                         install_signals, prepare_runtime, write_result)

KINDS = ("direct_flip", "direct_stay", "look_flip", "look_stay")
REQUIRED_CHECKS = ("restore_deterministic", "valid_physical", "unique_targets", "witness_replay", "evidence_available", "targets_separable")
PILOT_SCENES = ("FloorPlan1", "FloorPlan2", "FloorPlan201", "FloorPlan202", "FloorPlan25", "FloorPlan225")
COLLECTION_EDIT_DISTANCES_M = (0.05, 0.1, 0.25, 0.5, 1.0)


def predefined_split():
    rooms = {f"FloorPlan{i}": "kitchen" for i in range(1, 31)}
    rooms.update({f"FloorPlan{i}": "living_room" for i in range(201, 231)})
    return make_scene_split(rooms, [f"FloorPlan{i}" for i in (*range(25, 31), *range(225, 231))])


def classify(kind, validity, prefix):
    """Low-amplitude candidates never change exact fields or become passes."""
    if kind not in KINDS:
        raise ValueError("Unknown pair class")
    look = kind.startswith("look_")
    reasons = [name for name in REQUIRED_CHECKS if validity.get(name) is not True]
    reasons.extend(name for name, value in validity.items() if name not in REQUIRED_CHECKS and value is not True)
    strict = prefix.get("same_input_at_t_strict") is True
    if look and not strict:
        reasons.append("same_input_at_t")
    if not look and strict:
        reasons.append("direct_input_not_distinguishable")
    low_difference = (look and not strict and prefix.get("text_exact") is True
                      and 0 < prefix.get("prefix_max_abs_diff", 256) <= 3)
    if not reasons:
        return "passed", [], False
    if low_difference:
        # Preserve low-diff candidates even when they have independent failures.
        # Such failures remain listed and these rows cannot enter the pilot set.
        return "pending_review", reasons, True
    return "rejected", reasons, False


def check_configuration(config):
    for key in ("max_displacement_m", "min_evidence_pixels"):
        if config.get(key) is None and not config.get("collect_only"):
            raise ValueError(f"Missing required, externally chosen threshold: {key}")
    if config.get("max_displacement_m") is not None and (not isinstance(config["max_displacement_m"], (int, float)) or not math.isfinite(config["max_displacement_m"]) or config["max_displacement_m"] <= 0):
        raise ValueError("max_displacement_m must be finite and positive")
    if config.get("min_evidence_pixels") is not None and (type(config["min_evidence_pixels"]) is not int or config["min_evidence_pixels"] < 1):
        raise ValueError("min_evidence_pixels must be a positive integer")
    if not 12 <= config["viewpoints"] <= 20 or not 0 < config["timeout_seconds"] <= 4 * 3600 - 120:
        raise ValueError("Require 12..20 viewpoints and a timeout reserving cleanup within four GPU hours")
    if config["pairs_per_class"] < 20 or config["max_attempts"] < 1 or config["max_search_states"] < 1:
        raise ValueError("Require >=20 pairs/class and positive explicit search limits")
    if config["distance_tie_tolerance_m"] < 0 or not math.isfinite(config["distance_tie_tolerance_m"]):
        raise ValueError("Invalid relation tie tolerance")
    if not 0 < config["path_endpoint_tolerance_m"] <= 0.01:
        raise ValueError("Path endpoint tolerance must be positive and <=0.01m")


def task_candidates(objects):
    groups = {}
    for obj in objects:
        groups.setdefault(obj["objectType"], []).append(obj)
    candidates = []
    for target_type, targets in sorted(groups.items()):
        if len(targets) < 2 or not any(o.get("pickupable") or o.get("moveable") for o in targets):
            continue
        for anchor_type, anchors in sorted(groups.items()):
            if anchor_type == target_type or len(anchors) != 1:
                continue
            for relation in ("next to", "on", "in"):
                if relation != "next to" and not anchors[0].get("receptacle"):
                    continue
                candidates.append(RelationalTask(target_type, relation, anchor_type))
    return candidates


def propose_edit(world, task, rng, bound, collection_index=None):
    """One requested object edit; verify actual effects separately in Unity."""
    eligible = [o for o in world["objects"] if (o.get("pickupable") or o.get("moveable"))
                and o["objectType"] in (task.target_type, task.anchor_type)]
    eligible.sort(key=lambda o: o["name"])
    if not eligible:
        raise RuntimeError("No editable candidate or anchor")
    obj = rng.choice(eligible)
    if collection_index is None:
        radius = rng.uniform(0.25, 1.0) * bound
        angle = rng.uniform(0, 2 * math.pi)
    else:
        radius = COLLECTION_EDIT_DISTANCES_M[collection_index % len(COLLECTION_EDIT_DISTANCES_M)]
        angle = (collection_index // len(COLLECTION_EDIT_DISTANCES_M) % 8) * math.pi / 4
    # Horizontal translation preserves requested orientation; no surface-height
    # or AABB guess is treated as a legal placement check.
    requested = copy.deepcopy(world["poses"])
    edited = next(item for item in requested if item["objectName"] == obj["name"])
    edited["position"]["x"] += math.cos(angle) * radius
    edited["position"]["z"] += math.sin(angle) * radius
    other = copy.deepcopy(world)
    other.update(key=world["key"] + "_w1", poses=requested)
    return other, {"object_name": obj["name"], "object_id": obj["objectId"],
                   "requested_displacement_m": radius, "requested_pose": edited,
                   "kind": "horizontal_translation_one_object",
                   "collection_distance_is_not_acceptance_limit": collection_index is not None}


def serial_plan(plan):
    return {**{key: value for key, value in plan.items() if key not in ("actions", "states")},
            "actions": [action.text() for action in plan["actions"]],
            "states": [asdict(state) for state in plan["states"]]}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def review_image(path, row, pictures, target_ids):
    from PIL import Image, ImageDraw
    panels = []
    for title, observation, target in pictures:
        image = Image.fromarray(observation["marked"]).resize((448, 448))
        draw = ImageDraw.Draw(image)
        for mark in observation["marks"]:
            if mark["object_id"] == target:
                draw.rectangle(tuple(int(x * 448 / observation["rgb"].shape[0]) for x in mark["bounding_box"]), outline="lime", width=3)
        panel = Image.new("RGB", (448, 480), "white")
        panel.paste(image, (0, 32))
        ImageDraw.Draw(panel).text((5, 7), title, fill="black")
        panels.append(panel)
    sheet = Image.new("RGB", (448 * 2, 70 + 480 * ((len(panels) + 1) // 2)), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((5, 5), f"{row['attempt_id']} | {row['kind']} | {row['status']}", fill="black")
    draw.text((5, 25), row["instruction"], fill="black")
    draw.text((5, 45), "Green: correct target (only if visible); numbers are public marks.", fill="black")
    for index, image in enumerate(panels):
        sheet.paste(image, ((index % 2) * 448, 70 + (index // 2) * 480))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=90)
    return str(path)


def terminal_visible(observation, ids):
    return all(object_id in observation["masks"] and observation["masks"][object_id].any() for object_id in ids)


def choose_prefix(sim, worlds, task, targets, desired_kind):
    """Deterministic first qualifying state; all comparisons are real renders.

    There is exactly one initial frame (empty earlier visual history). Look
    receives the SAME observation action in both worlds, never twin target
    labels before evidence. A one-action common successor supplies evidence.
    """
    graph0, graph1 = [sim.graph(world) for world in worlds]
    relevant = sorted(set((*targets[0].candidate_ids, *targets[1].candidate_ids,
                           targets[0].anchor_id, targets[1].anchor_id)))
    look = desired_kind.startswith("look_")
    # One nonzero pixel is only a raw-candidate search rule in collection mode,
    # not the scientific evidence threshold. It is recorded as such in config.
    search_pixels = 1 if sim.config["collect_only"] else sim.config["min_evidence_pixels"]
    for node in sim.nodes:
        for heading in range(8):
            state = ViewState(node.viewpoint_id, heading)
            before = [sim.observe(world, state, task) for world in worlds]
            prefix = compare_inputs(*before, relevant)
            if look:
                if not prefix["text_exact"] or prefix["prefix_max_abs_diff"] > 3:
                    continue
                trans0 = {t.action.text(): t for t in sim.successors(worlds[0], state, before[0])}
                trans1 = {t.action.text(): t for t in sim.successors(worlds[1], state, before[1])}
                for text in sorted(trans0.keys() & trans1.keys()):
                    a, b = trans0[text], trans1[text]
                    if a.next_state != b.next_state:
                        continue
                    after = [sim.observe(world, a.next_state, task) for world in worlds]
                    evidence = compare_inputs(*after, relevant)
                    if evidence["relevant_difference_pixels"] < search_pixels:
                        continue
                    if not all(terminal_visible(obs, (*target.candidate_ids, target.anchor_id))
                               for obs, target in zip(after, targets)):
                        continue
                    return {"start": state, "before": before, "prefix": prefix,
                            "look_action": a.action, "look_state": a.next_state,
                            "after": after, "evidence": evidence, "relevant_ids": relevant}
            else:
                if prefix["same_input_at_t_strict"] or prefix["relevant_difference_pixels"] < search_pixels:
                    continue
                if not all(terminal_visible(obs, (*target.candidate_ids, target.anchor_id))
                           for obs, target in zip(before, targets)):
                    continue
                return {"start": state, "before": before, "prefix": prefix,
                        "look_action": None, "look_state": None, "after": before,
                        "evidence": prefix, "relevant_ids": relevant}
    raise RuntimeError("No qualifying rendered prefix and common one-action evidence view")


def restore_check(sim, world, state, task, directory):
    observations = [sim.observe(world, state, task) for _ in range(2)]
    comparison = compare_inputs(*observations, [])
    return {"passed": comparison["same_input_at_t_strict"], "comparison": comparison,
            "mode": "same_process_full_pose_identity_twice_no_reset",
            "artifacts": [save_observation(directory, f"restore_{i}", obs) for i, obs in enumerate(observations)]}


def build_attempt(sim, config, attempt_id, desired_kind, task, rng, directory, row):
    row.update({"attempt_id": attempt_id, "requested_kind": desired_kind, "kind": desired_kind,
           "scene": sim.scene, "split": "test" if sim.scene in config["scene_split"]["test_scenes"] else "train",
           "seed": config["seed"], "status": "rejected", "discard_reasons": [],
           "instruction": task.instruction({task.target_type: task.target_type, task.anchor_type: task.anchor_type}),
           "task": asdict(task), "same_input_at_t": None, "same_input_at_t_strict": None,
           "prefix_max_abs_diff": None, "history": "empty before single initial frame",
           "immediate_target_labels": None, "validity": {}})
    # Initialize a source from the current scene's fixed, saved seed layout.
    sim.apply_world(config["scene_world"])
    w0 = sim.world_from_event(attempt_id + "_w0")
    w1, edit = propose_edit(w0, task, rng, config["max_displacement_m"],
                           int(attempt_id.split("_")[-1]) if config["collect_only"] else None)
    row.update(w0=w0, w1=w1, edit=edit, viewpoints=[asdict(node) for node in sim.nodes])
    physical = [sim.physical_check(world, w0, edit["object_name"]) for world in (w0, w1)]
    row["physical_checks"] = physical
    targets, relation_evidence = [], []
    for world in (w0, w1):
        target, evidence = sim.relation(world, task)
        targets.append(target)
        relation_evidence.append(evidence)
    row["relation_evidence"] = relation_evidence
    row.update(target_w0=targets[0].target_id, target_w1=targets[1].target_id)
    if any(target.target_id is None for target in targets):
        raise AttemptRejected("target_not_unique_or_relation_unknown", row)
    flipped = targets[0].target_id != targets[1].target_id
    actual_kind = ("look_" if desired_kind.startswith("look_") else "direct_") + ("flip" if flipped else "stay")
    row["actual_target_change"] = "flip" if flipped else "stay"
    if actual_kind != desired_kind:
        if config["collect_only"]:
            row["kind"] = actual_kind
        else:
            raise AttemptRejected("edit_has_wrong_target_change", row)
    # Continue collection even for failed physics, preserving actual images and
    # low-diff candidates, but never turn the check into True afterwards.
    chosen = choose_prefix(sim, (w0, w1), task, targets, desired_kind)
    row.update(start=asdict(chosen["start"]), prefix_comparison=chosen["prefix"],
               same_input_at_t=chosen["prefix"]["same_input_at_t_strict"],
               same_input_at_t_strict=chosen["prefix"]["same_input_at_t_strict"],
               prefix_max_abs_diff=chosen["prefix"]["prefix_max_abs_diff"],
               evidence_pixels_after_look=chosen["evidence"]["relevant_difference_pixels"],
               evidence_comparison=chosen["evidence"])
    row["prefix_artifacts"] = [save_observation(directory, f"w{i}_prefix", obs) for i, obs in enumerate(chosen["before"])]
    row["after_observation_artifacts"] = [save_observation(directory, f"w{i}_after", obs) for i, obs in enumerate(chosen["after"])]
    row["common_observation_action"] = chosen["look_action"].text() if chosen["look_action"] else None
    row["after_observation_state"] = asdict(chosen["look_state"]) if chosen["look_state"] else None
    # Privileged targets remain task evaluation truth only. For Look the only
    # supplied immediate action label is the same common observation action.
    row["immediate_action_labels"] = ([row["common_observation_action"]] * 2 if chosen["look_action"] else None)
    row["restore_checks"] = [restore_check(sim, world, chosen["start"], task, directory / f"w{i}")
                              for i, world in enumerate((w0, w1))]
    witnesses = []
    for i, (world, target) in enumerate(zip((w0, w1), targets)):
        # For Look, force the same evidence-acquisition action before any
        # target-dependent decision. The final select still counts toward 12.
        planning_start = chosen["look_state"] or chosen["start"]
        plan = sim.plan(world, planning_start, task, target.target_id)
        if chosen["look_action"] and plan["found"]:
            plan["actions"] = [chosen["look_action"], *plan["actions"]]
            plan["states"] = [chosen["start"], *plan["states"]]
            if len(plan["actions"]) > 12:
                plan.update(found=False, reason="common_observation_plus_witness_exceeds_12")
        replay = sim.replay(world, chosen["start"], task, target.target_id, plan,
                            directory / f"w{i}", expected_prefix=chosen["before"][i],
                            expected_after=chosen["after"][i] if chosen["look_action"] else None)
        witnesses.append({"plan": serial_plan(plan), "replay": replay})
    row["witnesses"] = witnesses
    row["witness_actions_w0"] = witnesses[0]["replay"].get("actions", [])
    row["witness_actions_w1"] = witnesses[1]["replay"].get("actions", [])
    validity = {"restore_deterministic": all(x["passed"] for x in row["restore_checks"]),
                "valid_physical": (all(x["valid_physical"] is True for x in physical)
                                   if all(x["valid_physical"] is not None for x in physical) else None),
                "unique_targets": all(x.target_id is not None for x in targets),
                "witness_replay": all(x["replay"]["passed"] for x in witnesses),
                "evidence_available": (row["evidence_pixels_after_look"] >= config["min_evidence_pixels"]
                                       if config.get("min_evidence_pixels") is not None and not config["collect_only"] else None),
                "targets_separable": all(len(set(t.candidate_ids)) == len(t.candidate_ids) for t in targets)}
    # select resolves one visible unique label to one object ID, then compares
    # that ID to the current world's target; candidate success sets are disjoint.
    row.update(validity=validity, restore_deterministic=validity["restore_deterministic"],
               valid_physical=validity["valid_physical"], targets_separable=validity["targets_separable"])
    row["selection_success_rule"] = "one visible unique mark -> one objectId; selectedId == current unique targetId AND native DistanceToObject <= 1.5m"
    row["separability_proof"] = {
        "candidate_ids_w0": list(targets[0].candidate_ids),
        "candidate_ids_w1": list(targets[1].candidate_ids),
        "mechanism": "Success for candidate c includes selected_object_id == c; distinct candidate IDs cannot both equal the same selected ID.",
        "mark_mapping_validation": "visible_marks rejects overlapping marked masks; selection_action rejects duplicate labels and requires a nonempty candidate mask",
        "replayed_select_records": [witness["replay"].get("actions", [])[-1:]
                                     for witness in witnesses],
        "does_not_infer_separability_from_geometry": True}
    if config["collect_only"]:
        status, reasons, pending = "collected", [], False
        row["threshold_classification"] = "not_performed_collect_only"
        row["independent_check_failures"] = [name for name, value in validity.items() if value is False]
    else:
        status, reasons, pending = classify(desired_kind, validity, chosen["prefix"])
    row.update(status=status, discard_reasons=reasons, low_diff_pending_review=pending,
               pending_has_other_failures=bool(pending and any(r != "same_input_at_t" for r in reasons)))
    row["geometry_graphs"] = [{"context": graph.context_key, "edges": [asdict(e) for e in graph.edges],
                                "rejected_edges": graph.rejected_edges} for graph in sim.graph_cache.values()]
    pictures = [(f"w{i} initial", chosen["before"][i], targets[i].target_id) for i in range(2)]
    if chosen["look_action"]:
        pictures += [(f"w{i} after {chosen['look_action'].text()}", chosen["after"][i], targets[i].target_id) for i in range(2)]
    return row, pictures


class AttemptRejected(RuntimeError):
    def __init__(self, reason, row):
        super().__init__(reason)
        self.row = row


def compact_record(row, raw_path):
    """Keep bounded portable bookkeeping; full metadata/arrays stay on scratch."""
    compact = copy.deepcopy(row)
    for key in ("w0", "w1"):
        if key in compact:
            compact[key].pop("objects", None)
    for check in compact.get("physical_checks", []):
        check.pop("realized_objects", None)
    compact.pop("geometry_graphs", None)
    raw_bytes = raw_path.read_bytes()
    compact["raw_attempt_evidence"] = {"path": str(raw_path), "bytes": len(raw_bytes),
                                      "sha256": hashlib.sha256(raw_bytes).hexdigest()}
    return compact


def metrics_from_rows(rows, config, scene_failures, status):
    counts = {kind: {name: 0 for name in ("passed", "pending_review", "rejected", "collected")} for kind in KINDS}
    reasons = {kind: Counter() for kind in KINDS}
    times = []
    look_observed, look_strict = 0, 0
    pending_eligible = {kind: 0 for kind in KINDS}
    for row in rows:
        kind = row["kind"]
        counts[kind][row["status"]] += 1
        reasons[kind].update(row.get("discard_reasons", []))
        if row.get("generation_seconds") is not None:
            times.append(row["generation_seconds"])
        if kind.startswith("look_") and row.get("same_input_at_t_strict") is not None:
            look_observed += 1
            look_strict += int(row["same_input_at_t_strict"])
        if row["status"] == "pending_review" and not row.get("pending_has_other_failures", True):
            pending_eligible[kind] += 1
    passed = all(counts[k]["passed"] >= config["pairs_per_class"] for k in KINDS)
    a, b = counts["look_flip"]["passed"], counts["look_stay"]["passed"]
    ratio = a / b if b else None
    return {"status": status, "attempts": len(rows), "counts_by_class": counts,
        "discard_reason_counts": {k: dict(v) for k, v in reasons.items()},
        "pending_prefix_only_by_class": pending_eligible,
        "mean_pair_generation_seconds": sum(times) / len(times) if times else None,
        "generation_time_denominator": "all attempted pairs with recorded duration, including rejected attempts; not amortized successful-pair cost",
        "generation_time_observed_attempts": len(times),
        "look_prefix_comparisons": look_observed, "look_strict_prefix_count": look_strict,
        "look_strict_prefix_fraction": look_strict / look_observed if look_observed else None,
        "look_passed_flip_stay_ratio": ratio,
        "acceptance_passed": False if config["collect_only"] else bool(passed and ratio is not None and 0.8 <= ratio <= 1.25),
        "collect_only": config["collect_only"],
        "review_images": [row["review_image"] for row in rows if row.get("review_image")],
        "scene_failures": scene_failures,
        "all_attempts_path": config["output_paths"]["all"],
        "passed_pairs_path": config["output_paths"].get("passed"),
        "pending_pairs_path": config["output_paths"].get("pending_review"),
        "raw_candidates_path": config["output_paths"].get("collected")}


def generator_worker(config, connection):
    rows, scene_failures = [], []
    data = ROOT / "data" / "world_pairs"
    data.mkdir(parents=True, exist_ok=True)
    files = {key: ROOT / path for key, path in config["output_paths"].items()}
    # Never overwrite a prior pilot. Parent must choose a fresh run workspace or
    # explicitly archive existing outputs via the usual provenance workflow.
    for path in files.values():
        if path.exists():
            raise RuntimeError(f"Output already exists; preserved: {path}")
    handles = {key: path.open("x") for key, path in files.items()}
    reviewed = Counter()
    rng = random.Random(config["seed"])
    reserve_bytes = 2 * 1024 * 1024

    def publish(status="running"):
        metrics = metrics_from_rows(rows, config, scene_failures, status)
        connection.send(("progress", metrics))
        return metrics

    try:
        for scene_index, scene in enumerate(config["pilot_scenes"]):
            if len(rows) >= config["max_attempts"] or time.time() >= config["deadline"]:
                break
            scene_config = {**config, "scene": scene}
            try:
                with controller_session(scene_config, f"scene_{scene}", segmentation=True, connection=connection) as controller:
                    build = getattr(getattr(controller, "_build", None), "commit_id", None)
                    if build != BUILD:
                        raise RuntimeError(f"Expected reviewed build {BUILD}, found {build}")
                    sim = PairSimulator(controller, scene_config)
                    # One controller per scene; no reset occurs during any pair.
                    # Seeded source is fixed before attempts, never chosen by results.
                    sim.action("InitialRandomSpawn", randomSeed=config["seed"],
                               forceVisible=False, placeStationary=True, numPlacementAttempts=5)
                    scene_world = sim.world_from_event(scene + "_source")
                    scene_config["scene_world"] = scene_world
                    config_for_pair = {**config, "scene_world": scene_world}
                    scene_dir = Path(config["raw_evidence_dir"]) / scene
                    write_json(scene_dir / "source.json", {"world": scene_world, "random_seed": config["seed"]})
                    from PIL import Image
                    Image.fromarray(controller.last_event.frame).save(scene_dir / "source_rgb.png")
                    node_evidence = sim.sample_nodes(config["seed"])
                    write_json(scene_dir / "navigation_sample.json", node_evidence)
                    tasks = task_candidates(scene_world["objects"])
                    if not tasks:
                        raise RuntimeError("No relation tasks with duplicate candidate category and unique anchor")
                    # Fixed per-scene quota avoids exhausting all attempts before
                    # reaching the predeclared held-out scenes.
                    quota = math.ceil(config["max_attempts"] / len(config["pilot_scenes"]))
                    for local in range(quota):
                        if len(rows) >= config["max_attempts"]:
                            break
                        if any(handle.tell() > config["journal_max_bytes"] - reserve_bytes for handle in handles.values()):
                            return publish("journal_size_limit")
                        sim.check_deadline()
                        counts = publish()["counts_by_class"]
                        active = [kind for kind in KINDS if counts[kind]["collected" if config["collect_only"] else "passed"] < config["pairs_per_class"]]
                        if not active:
                            return publish("completed")
                        kind = active[len(rows) % len(active)]
                        task = tasks[rng.randrange(len(tasks))]
                        attempt_id = f"pair_{len(rows):05d}"
                        directory = Path(config["raw_evidence_dir"]) / attempt_id
                        directory.mkdir(parents=True, exist_ok=True)
                        started = time.monotonic()
                        sim.query_log, sim.graph_cache, sim.observation_cache = [], {}, {}
                        row = {"attempt_id": attempt_id, "kind": kind, "scene": scene, "status": "rejected",
                               "discard_reasons": [], "task": asdict(task)}
                        pictures = []
                        try:
                            row, pictures = build_attempt(sim, config_for_pair, attempt_id, kind, task, rng, directory, row)
                        except AttemptRejected as error:
                            row = error.row
                            row["discard_reasons"] = [str(error)]
                        except (TimeoutError, KeyboardInterrupt, SystemExit):
                            row["discard_reasons"] = ["deadline_or_interrupt"]
                            raise
                        except Exception as error:
                            row["discard_reasons"] = [f"{type(error).__name__}: {error}"]
                            traceback.print_exc()
                        finally:
                            # Save a real last frame even for a rejected attempt;
                            # it is labeled last-observed, never invented as w0/w1.
                            try:
                                from PIL import Image
                                Image.fromarray(controller.last_event.frame).save(directory / "last_observed.png")
                                row["last_observed_image"] = str(directory / "last_observed.png")
                            except Exception as error:
                                row["last_frame_error"] = str(error)
                            row["generation_seconds"] = time.monotonic() - started
                            if pictures and row["status"] in ("passed", "pending_review", "collected") and reviewed[row["kind"]] < 10:
                                review_path = (ROOT / "logs" / "review" / "pilot" / config["name"] /
                                               f"{row['kind']}_{reviewed[row['kind']]:02d}_{attempt_id}.jpg")
                                try:
                                    row["review_image"] = review_image(review_path, row, pictures,
                                                                        [row["target_w0"], row["target_w1"]])
                                    # Only the deterministic first ten per class
                                    # export small key PNGs to home. Arrays and
                                    # complete replay evidence remain on scratch.
                                    exported = []
                                    for group in ("prefix_artifacts", "after_observation_artifacts"):
                                        for item in row.get(group, []):
                                            source_path = Path(item["rgb"])
                                            destination = review_path.parent / (attempt_id + "_" + source_path.name)
                                            shutil.copyfile(source_path, destination)
                                            exported.append({"path": str(destination),
                                                "sha256": hashlib.sha256(destination.read_bytes()).hexdigest()})
                                    row["exported_key_pngs"] = exported
                                    reviewed[row["kind"]] += 1
                                except Exception as error:
                                    row["review_image_error"] = str(error)
                            row["query_log"] = str(directory / "queries.json")
                            write_json(directory / "queries.json", sim.query_log)
                            row["query_log_sha256"] = hashlib.sha256((directory / "queries.json").read_bytes()).hexdigest()
                            write_json(directory / "attempt.json", row)
                            row = compact_record(row, directory / "attempt.json")
                            encoded = json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
                            if len(encoded.encode()) > reserve_bytes:
                                row = {"attempt_id": attempt_id, "kind": row["kind"], "scene": scene,
                                       "status": "rejected", "discard_reasons": ["journal_record_size_limit"],
                                       "raw_attempt_evidence": row["raw_attempt_evidence"],
                                       "generation_seconds": row["generation_seconds"]}
                                encoded = json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
                            rows.append(row)
                            handles["all"].write(encoded)
                            handles["all"].flush()
                            if row["status"] in ("passed", "pending_review", "collected"):
                                handles[row["status"]].write(encoded)
                                handles[row["status"]].flush()
                            print("T01_ATTEMPT_JSON: " + json.dumps({k: row.get(k) for k in
                                ("attempt_id", "kind", "status", "discard_reasons", "same_input_at_t_strict", "prefix_max_abs_diff")}), flush=True)
                            publish()
            except (TimeoutError, KeyboardInterrupt, SystemExit):
                raise
            except Exception as error:
                scene_failures.append({"scene": scene, "error": f"{type(error).__name__}: {error}",
                                       "evidence_directory": str(Path(config["raw_evidence_dir"]) / scene)})
                publish()
        return publish("completed_search")
    finally:
        for handle in handles.values():
            handle.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="pilot_pairs")
    parser.add_argument("--max-displacement-m", type=float)
    parser.add_argument("--min-evidence-pixels", type=int)
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pairs-per-class", type=int, default=20)
    parser.add_argument("--max-attempts", type=int, default=240)
    parser.add_argument("--timeout-seconds", type=float, default=4 * 3600 - 120)
    parser.add_argument("--max-search-states", type=int, default=256)
    parser.add_argument("--viewpoints", type=int, default=12)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--distance-tie-tolerance-m", type=float, default=0.0)
    parser.add_argument("--path-endpoint-tolerance-m", type=float, default=0.01)
    parser.add_argument("--enclosed-receptacle-types", nargs="*", default=["Cabinet", "Drawer", "Fridge", "Microwave", "Safe"])
    args = parser.parse_args(argv)
    config = vars(args)
    output_paths = ({"all": f"data/world_pairs/{args.name}_raw_attempts.jsonl",
                     "collected": f"data/world_pairs/{args.name}_raw_candidates.jsonl"} if args.collect_only else
                    {"all": "data/world_pairs/pilot_attempts.jsonl", "passed": "data/world_pairs/pilot.jsonl",
                     "pending_review": "data/world_pairs/pilot_pending_review.jsonl"})
    config.update(scene=PILOT_SCENES[0], pilot_scenes=list(PILOT_SCENES), scene_split=predefined_split(),
                  output_paths=output_paths, collection_edit_distances_m=list(COLLECTION_EDIT_DISTANCES_M),
                  collection_angles_degrees=list(range(0, 360, 45)),
                  collection_candidate_search_min_nonzero_pixels=1,
                  journal_max_bytes=32 * 1024 * 1024,
                  grid_size=0.1, visibility_distance=10, field_of_view=100, max_open_containers=1,
                  pinned_unity_build=BUILD,
                  threshold_source=("not applied in collect-only; requires later external confirmation" if args.collect_only
                                    else "explicit CLI chosen before measurement"),
                  quality="Ultra (ai2thor 5.0.0 default)",
                  scene_order_policy="fixed declared train/test scenes; no replacement after failures",
                  backend="live, no cached prefix reuse", maximum_witness_actions_including_select=12,
                  physical_claim="native primitive-collider query scope only; no AABB nonpenetration inference",
                  selection_distance_method="AI2THOR DistanceToObject horizontal collider-bound distance <=1.5m",
                  notes=["Prefix includes unmarked RGB, marked RGB, and all public instruction/navigation text.",
                         "One initial frame, empty prior history; Look immediate labels are one shared observation action.",
                         "Pending low-diff rows remain excluded from pilot.jsonl and acceptance counts.",
                         "No reset between w0 and w1 or within their restoration/replay checks.",
                         "Full raw images/masks/replays/metadata remain on scratch; retention rules unknown.",
                         "Home journals are capped at 32MiB each, with at most 240 attempts by default; oversized records point to retained scratch evidence.",
                         "First ten retained attempts per class, in generation order, form the maximum forty review images."])
    metrics, worker = {"status": "failed_before_generation", "acceptance_passed": False}, None
    install_signals()
    try:
        check_configuration(config)
        if args.name != "pilot_pairs" and not (args.collect_only and args.name.startswith("pilot_pairs_")):
            raise ValueError("Use the registered run name pilot_pairs; separate reruns require reviewed output isolation")
        config["deadline"] = deadline_after(args.timeout_seconds)
        config["runtime"] = prepare_runtime(args.name)
        config["raw_evidence_dir"] = str(Path(config["runtime"]["scratch"]) / "t01_raw" /
                                         args.name / config["runtime"]["pbs_jobid"])
        Path(config["raw_evidence_dir"]).mkdir(parents=True, exist_ok=False)
        worker = Worker(generator_worker, config)
        while True:
            worker.receive()
            if worker.latest:
                metrics = copy.deepcopy(worker.latest)
            if worker.result is not None:
                metrics = worker.result
                break
            if worker.error:
                metrics.update(status="worker_failed", error=worker.error, acceptance_passed=False)
                break
            if time.time() >= config["deadline"]:
                raise TimeoutError("T01 parent deadline reached")
            time.sleep(0.1)
    except BaseException as error:
        metrics.update(status="incomplete", error=f"{type(error).__name__}: {error}", acceptance_passed=False)
        traceback.print_exc()
    finally:
        if worker:
            errors = worker.stop()
            if errors:
                metrics.update(cleanup_errors=errors, acceptance_passed=False)
        if metrics.get("attempts", 0) > 0:
            metrics["output_files"] = [{"path": path, "sha256": hashlib.sha256((ROOT / path).read_bytes()).hexdigest()}
                                       for path in config["output_paths"].values() if (ROOT / path).exists()]
        write_result(args.name, metrics, config)
    return 0 if metrics.get("acceptance_passed") or (args.collect_only and metrics.get("attempts", 0) > 0
            and metrics.get("status") in ("completed", "completed_search") and not metrics.get("cleanup_errors")) else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Pure CPU T01v2 measurements; no simulator calls, writes, or experiment runs.

Object identity is the stable Unity name. Metadata centres define the declared
3-D distance. D1 public-prefix reuse never changes the independently measured
state-level restore_deterministic measurement or certifies all Unity state.
"""
import copy
import math
import re

import numpy as np

from .pair_approved_policy import indexed_objects, position_distance, rotation_delta_degrees

DISTANCE_METHOD = "3d_euclidean_ai2thor_metadata_position_centres"
POLICY = {
    "id": "t01v2_constructed_closest_anchor_20261108",
    "distance_method": DISTANCE_METHOD, "closest_separation_ratio": 1.5,
    "visible_pixels_at_255": 250, "evidence_pixels_at_255": 250,
    "evidence_channel_difference_exclusive": 10, "look_prefix_max_channel_difference": 3,
    "edited_position_error_max_m": 0.02, "other_position_change_max_m": 0.01,
    "select_distance_max_m": 1.5, "restore_is_independent_measurement": True,
    "restore_acceptance_decision": "METRICS revision 6d234b8: editable positions within 1cm and exact agent pose; RGB difference diagnostic",
}


def area_passes(pixels, height, width):
    if any(type(x) is not int for x in (pixels, height, width)) or min(height, width) <= 0 or not 0 <= pixels <= height * width:
        raise ValueError("Invalid integer mask count or image dimensions")
    return pixels * 255 * 255 >= 250 * height * width


def closest_target(objects, candidate_names, anchor_name):
    """Resolve the complete category-only instruction, including zero-distance ties."""
    indexed = indexed_objects(objects)
    names = tuple(candidate_names)
    reasons = []
    if len(names) != 2 or len(set(names)) != 2 or anchor_name in names:
        raise ValueError("Require two distinct candidates and a distinct anchor")
    if not set((*names, anchor_name)) <= set(indexed):
        raise ValueError("Candidate or anchor missing from full inventory")
    a, b, x = (indexed[name] for name in (*names, anchor_name))
    if a.get("objectType") != b.get("objectType") or not a.get("objectType"):
        reasons.append("candidate_categories_differ")
    if x.get("objectType") == a.get("objectType") or not x.get("objectType"):
        reasons.append("anchor_category_not_distinct")
    if not all(obj.get("pickupable") is True for obj in (a, b, x)):
        reasons.append("candidate_or_anchor_not_pickupable")
    if {obj["name"] for obj in objects if obj.get("objectType") == a.get("objectType")} != set(names):
        reasons.append("unlisted_same_category_candidate")
    if [obj["name"] for obj in objects if obj.get("objectType") == x.get("objectType")] != [anchor_name]:
        reasons.append("category_only_anchor_not_unique")
    distances = {name: position_distance(indexed[name]["position"], x["position"]) for name in names}
    ordered = sorted(names, key=lambda name: (distances[name], name))
    near, far = (distances[name] for name in ordered)
    if near == far:
        reasons.append("nearest_target_tie")
    if far < 1.5 * near:
        reasons.append("closest_ratio_below_1_5")
    return {"passed": not reasons, "target_name": ordered[0] if not reasons else None,
            "candidate_names": list(names), "anchor_name": anchor_name, "distances_m": distances,
            "distance_method": DISTANCE_METHOD, "reasons": reasons}


def _rgb(value):
    value = np.asarray(value)
    if value.dtype != np.uint8 or value.ndim != 3 or value.shape[2] != 3 or min(value.shape[:2]) <= 0:
        raise ValueError("Expected nonempty HxWx3 uint8 RGB")
    return value


def _mask(observation, name):
    shape = _rgb(observation["rgb"]).shape[:2]
    masks = observation["masks_by_name"]  # An absent mapping is unknown, not invisible.
    if name not in masks:
        return np.zeros(shape, dtype=bool)
    mask = np.asarray(masks[name])
    if mask.dtype != np.bool_ or mask.shape != shape:
        raise ValueError("Instance mask dtype/shape mismatch")
    return mask


def public_copy(observation):
    """Only public fields; never carry privileged object names into model input."""
    return {key: copy.deepcopy(observation[key]) for key in
            ("rgb", "marked", "input_text", "marks", "history")}


def public_exact(first, second):
    return (np.array_equal(_rgb(first["rgb"]), _rgb(second["rgb"]))
            and np.array_equal(_rgb(first["marked"]), _rgb(second["marked"]))
            and all(first[key] == second[key] for key in ("input_text", "marks", "history")))


def mark_observation(capture, candidate_names, instruction, *, public_context="", history=()):
    """Deterministic spatial marks on all visible candidates, never distance filtered."""
    from PIL import Image, ImageDraw
    result = dict(capture)
    rgb = _rgb(capture["rgb"])
    if len(set(candidate_names)) != len(candidate_names):
        raise ValueError("Duplicate candidate name")
    occupied = np.zeros(rgb.shape[:2], dtype=bool)
    entries = []
    for name in candidate_names:
        mask = _mask(capture, name)
        if np.any(occupied & mask):
            raise ValueError("Candidate instance masks overlap")
        occupied |= mask
        ys, xs = np.nonzero(mask)
        if not len(xs):
            continue
        pixels = sorted(zip(xs.tolist(), ys.tolist()), key=lambda p: (p[1], p[0]))
        count, sx, sy = len(pixels), sum(xs.tolist()), sum(ys.tolist())
        anchor = min(pixels, key=lambda p: ((p[0] * count - sx) ** 2 + (p[1] * count - sy) ** 2, p[1], p[0]))
        box = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
        entries.append(((box[1], box[0], box[3], box[2], pixels), name, box, anchor))
    entries.sort(key=lambda row: row[0])
    marked = Image.fromarray(rgb)
    draw = ImageDraw.Draw(marked)
    marks, labels = [], {}
    for label, (_, name, box, anchor) in enumerate(entries, 1):
        labels[name] = label
        marks.append({"label": label, "bounding_box": box, "anchor_pixel": list(anchor)})
        draw.rectangle(draw.textbbox(anchor, str(label)), fill="white")
        draw.text(anchor, str(label), fill="black")
    result.update(rgb=rgb.copy(), marked=np.asarray(marked).copy(), marks=marks,
                  mark_labels_by_name=labels, history=copy.deepcopy(list(history)),
                  input_text=instruction + ("\n" + public_context if public_context else ""))
    return result


def compare_view(first, second, relevant_names):
    left, right = _rgb(first["rgb"]), _rgb(second["rgb"])
    if left.shape != right.shape:
        raise ValueError("Paired view dimensions differ")
    delta = np.abs(left.astype(np.int16) - right.astype(np.int16))
    union = np.zeros(left.shape[:2], dtype=bool)
    for observation in (first, second):
        for name in relevant_names:
            union |= _mask(observation, name)
    count = int(np.count_nonzero(union & np.any(delta > 10, axis=2)))
    marked_delta = np.abs(_rgb(first["marked"]).astype(np.int16) - _rgb(second["marked"]).astype(np.int16))
    if marked_delta.shape != delta.shape:
        raise ValueError("Marked RGB dimensions differ")
    policy_count = int(np.count_nonzero(union & np.any(marked_delta > 10, axis=2)))
    return {"rgb_exact": bool(np.array_equal(left, right)), "public_input_exact": public_exact(first, second),
            "prefix_max_abs_diff": int(delta.max()), "relevant_mask_union_pixels": int(union.sum()),
            "evidence_pixels": count, "evidence_passed": area_passes(count, *left.shape[:2]),
            "marked_evidence_pixels": policy_count,
            "marked_evidence_passed": area_passes(policy_count, *left.shape[:2]),
            "resolution": list(left.shape[:2]), "evidence_channel_difference_exclusive": 10}


def prefix_check(prefix_w0, prefix_w1, candidate_names, anchor_name, *,
                 shared_prefix_w0=None, shared_prefix_w1=None, same_process, agent_unchanged):
    """Measure independent captures, then verify the actual supplied shared inputs.

    Missing shared inputs cannot certify same_input_at_t, even if raw frames
    happened to match. Candidate A/B remain visible; only edited anchor X hides.
    """
    if not prefix_w0 or len(prefix_w0) != len(prefix_w1):
        raise ValueError("Require equal nonempty prefix sequences")
    if len(candidate_names) != 2 or len(set(candidate_names)) != 2 or anchor_name in candidate_names:
        raise ValueError("Require two candidates and distinct edited anchor")
    rows, reasons = [], []
    if same_process is not True or agent_unchanged is not True:
        reasons.append("prefix_construction_process_or_agent_not_verified")
    candidate_visible = True
    for left, right in zip(prefix_w0, prefix_w1):
        row = compare_view(left, right, (*candidate_names, anchor_name))
        counts = [{name: int(_mask(obs, name).sum()) for name in (*candidate_names, anchor_name)} for obs in (left, right)]
        visible = all(area_passes(count[name], *_rgb(obs["rgb"]).shape[:2])
                      and name in obs["mark_labels_by_name"] for obs, count in zip((left, right), counts) for name in candidate_names)
        for obs in (left, right):
            labels = [obs["mark_labels_by_name"].get(name) for name in candidate_names]
            public_labels = [mark.get("label") for mark in obs["marks"]]
            visible = visible and (all(type(label) is int and label > 0 for label in labels)
                                   and len(set(labels)) == 2 and all(public_labels.count(label) == 1 for label in labels))
        candidate_visible &= visible
        labels_match = (left["mark_labels_by_name"] == right["mark_labels_by_name"])
        context_match = all(left[key] == right[key] for key in ("input_text", "history"))
        row.update(mask_pixels_w0=counts[0], mask_pixels_w1=counts[1], candidates_visible_and_marked=visible,
                   label_bindings_exact=labels_match, public_context_exact=context_match)
        if not visible:
            reasons.append("candidate_prefix_visibility_or_marks")
        if row["prefix_max_abs_diff"] > 3:
            reasons.append("prefix_rgb_difference_above_3")
        if counts[0][anchor_name] != 0 or counts[1][anchor_name] != 0:
            reasons.append("edited_anchor_visible_in_prefix")
        if not labels_match or not context_match:
            reasons.append("prefix_public_context_or_label_bindings_differ")
        rows.append(row)
    render_qualified = not reasons
    shared_verified = False
    if shared_prefix_w0 is not None and shared_prefix_w1 is not None:
        shared_verified = (len(shared_prefix_w0) == len(prefix_w0) == len(shared_prefix_w1)
                           and all(public_exact(a, b) and public_exact(a, source)
                                   for a, b, source in zip(shared_prefix_w0, shared_prefix_w1, prefix_w0)))
    return {"render_qualified_for_shared_prefix": render_qualified,
            "construction_context_verified": same_process is True and agent_unchanged is True,
            "shared_prefix_verified": shared_verified, "same_input_at_t": render_qualified and shared_verified,
            "native_public_prefix_exact": all(row["public_input_exact"] for row in rows),
            "candidates_visible_and_marked": candidate_visible,
            "prefix_max_abs_diff": max(row["prefix_max_abs_diff"] for row in rows),
            "prefix_edit_mask_pixels": [{"w0": row["mask_pixels_w0"][anchor_name], "w1": row["mask_pixels_w1"][anchor_name]} for row in rows],
            "frames": rows, "reasons": sorted(set(reasons + ([] if shared_verified else ["shared_prefix_not_verified"])))}


def physical_check(before_objects, requested_poses, realized_objects, edited_names, *,
                   native_is_scene_at_rest, placement_evidence_by_name):
    """Approved settling measurement. Rotations are recorded without a threshold."""
    before, after = indexed_objects(before_objects), indexed_objects(realized_objects)
    edited = set(edited_names)
    requested = {pose["objectName"]: pose for pose in requested_poses}
    if len(requested) != len(requested_poses) or not edited or not edited <= set(before):
        raise ValueError("Duplicate poses or missing edited identity")
    editable = {name for name, obj in before.items() if obj.get("moveable") is True or obj.get("pickupable") is True}
    reasons = []
    if set(requested) != editable:
        reasons.append("incomplete_movable_pickupable_pose_list")
    if set(before) != set(after):
        reasons.append("object_inventory_changed")
    if native_is_scene_at_rest is not True:
        reasons.append("scene_not_verified_at_rest")
    changes = {}
    for name in sorted(set(before) & set(after)):
        changes[name] = {"position_change_m": position_distance(before[name]["position"], after[name]["position"]),
                         "rotation_component_change_degrees": rotation_delta_degrees(before[name]["rotation"], after[name]["rotation"])}
        if name not in edited and changes[name]["position_change_m"] > 0.01:
            reasons.append("other_object_position_change_above_0_01m:" + name)
    edits = {}
    for name in sorted(edited):
        if name not in requested or name not in after:
            reasons.append("edited_object_missing:" + name)
            continue
        proof = placement_evidence_by_name.get(name) or {}
        receptacle = proof.get("receptacle_name")
        sample = proof.get("selected_position")
        sampled = (proof.get("action") == "GetSpawnCoordinatesAboveReceptacle" and proof.get("success") is True
                   and isinstance(sample, dict) and sample in (proof.get("coordinates") or []))
        geometry = proof.get("geometry")
        if geometry is None:
            # An untransformed native point is only a candidate, not placement success.
            pivot_verified = sample == requested[name]["position"]
        else:
            arguments = geometry.get("placement_arguments") or {}
            pivot_verified = (geometry.get("method") == "native_PlaceObjectAtPoint_forceKinematic"
                and geometry.get("object_name") == name and geometry.get("placement_action") == "PlaceObjectAtPoint"
                and geometry.get("placement_success") is True and arguments.get("forceKinematic") is True
                and isinstance(arguments.get("objectId"), str) and bool(arguments["objectId"])
                and arguments.get("position") == sample
                and arguments.get("rotation") == geometry.get("source_rotation") == requested[name]["rotation"]
                and geometry.get("returned_position") == proof.get("requested_position") == requested[name]["position"]
                and geometry.get("captured_before_complete_pose_set_and_settling") is True
                and bool(geometry.get("placement_reference")))
        sampled = sampled and pivot_verified
        parent_ok = (receptacle in after and after[receptacle].get("receptacle") is True
                     and after[receptacle].get("objectId") is not None
                     and after[receptacle]["objectId"] in (after[name].get("parentReceptacles") or []))
        error = position_distance(requested[name]["position"], after[name]["position"])
        edits[name] = {"position_error_m": error, "legal_sample_verified": sampled, "parent_receptacle_verified": parent_ok,
                       "rotation_error_components_degrees": rotation_delta_degrees(requested[name]["rotation"], after[name]["rotation"])}
        if error > 0.02:
            reasons.append("edited_position_error_above_0_02m:" + name)
        if not sampled or not parent_ok:
            reasons.append("legal_placement_or_parent_unverified:" + name)
    return {"valid_physical": not reasons, "native_is_scene_at_rest": native_is_scene_at_rest,
            "full_pose_list_verified": set(requested) == editable, "edited_names": sorted(edited),
            "edits": edits, "all_object_changes": changes, "reasons": reasons,
            "measurement_scope": "approved legal-placement plus settling response; not universal mesh-penetration proof"}


def restore_state_check(snapshot_objects, restored_objects, snapshot_agent, restored_agent):
    """6d234b8 state restoration gate; RGB is deliberately not an input.

    The backend invokes this immediately after snapshot restore, before witness
    actions. Object Euler changes are recorded; only object position is bounded.
    """
    before, after = indexed_objects(snapshot_objects), indexed_objects(restored_objects)
    editable = lambda rows: {name for name, row in rows.items() if row.get("moveable") is True or row.get("pickupable") is True}
    expected, observed = editable(before), editable(after)
    reasons, changes = [], {}
    if not expected or expected != observed:
        reasons.append("restored_movable_pickupable_inventory_mismatch_or_empty")
    for name in sorted(expected & observed):
        changes[name] = {"position_error_m": position_distance(before[name]["position"], after[name]["position"]),
                         "rotation_component_change_degrees": rotation_delta_degrees(before[name]["rotation"], after[name]["rotation"])}
        if changes[name]["position_error_m"] > 0.01:
            reasons.append("restored_position_error_above_0_01m:" + name)
    def canonical_agent(agent):
        # thor_checks.agent_pose uses TeleportFull's horizon/standing names;
        # native metadata uses cameraHorizon/isStanding. Neither may be absent.
        value = {"position": agent.get("position"), "rotation": agent.get("rotation"),
                 "horizon": agent.get("horizon", agent.get("cameraHorizon")),
                 "standing": agent.get("standing", agent.get("isStanding"))}
        if any(old in agent and new in agent and agent[old] != agent[new]
               for old, new in (("horizon", "cameraHorizon"), ("standing", "isStanding"))):
            return None
        for key in ("position", "rotation"):
            if not isinstance(value[key], dict) or any(type(value[key].get(axis)) not in (int, float)
                    or not math.isfinite(value[key][axis]) for axis in ("x", "y", "z")):
                return None
        if type(value["standing"]) is not bool or type(value["horizon"]) not in (int, float) or not math.isfinite(value["horizon"]):
            return None
        return value
    fields = ("position", "rotation", "horizon", "standing")
    old_agent, new_agent = canonical_agent(snapshot_agent), canonical_agent(restored_agent)
    agent_exact = old_agent is not None and old_agent == new_agent
    if not agent_exact:
        reasons.append("restored_agent_pose_not_exact_or_unknown")
    return {"restore_deterministic": not reasons, "passed": not reasons,
            "all_movable_position_errors": changes, "agent_pose_exact": agent_exact,
            "agent_pose_fields": list(fields), "snapshot_agent": copy.deepcopy(snapshot_agent),
            "restored_agent": copy.deepcopy(restored_agent), "reasons": reasons,
            "scope": "reported editable positions and agent pose, not native RGB or complete Unity state"}


def single_edit_check(poses_w0, poses_w1, anchor_name):
    first = {row["objectName"]: row for row in poses_w0}
    second = {row["objectName"]: row for row in poses_w1}
    if len(first) != len(poses_w0) or len(second) != len(poses_w1):
        raise ValueError("Duplicate requested pose names")
    changed = sorted(name for name in set(first) & set(second) if first[name] != second[name])
    passed = (set(first) == set(second) and changed == [anchor_name]
              and first[anchor_name]["position"] != second[anchor_name]["position"]
              and first[anchor_name]["rotation"] == second[anchor_name]["rotation"])
    return {"passed": passed, "changed_requested_names": changed,
            "scope": "requested edits; realized settling drift checked independently"}


def witness_check(record, target_name, *, max_actions=12):
    """Check actual new-process replay evidence; graph plans alone cannot pass.

    Actions are public Action.text() strings; successes are per executed action.
    The backend records actual selected name, mask count and centre distance.
    """
    planned, executed = record.get("planned_actions"), record.get("executed_actions")
    reasons = []
    if type(max_actions) is not int or max_actions < 1 or max_actions > 12:
        raise ValueError("Witness budget must include select and be at most 12")
    actions_ok = (isinstance(planned, list) and planned == executed and 1 <= len(planned) <= max_actions
                  and all(isinstance(action, str) for action in planned)
                  and planned[-1].startswith("select ")
                  and all(not action.startswith("select ") for action in planned[:-1]))
    if not actions_ok:
        reasons.append("executed_witness_missing_mismatched_or_over_budget")
    successes = record.get("action_successes")
    if (not isinstance(executed, list) or not isinstance(successes, list) or len(successes) != len(executed)
            or not all(value is True for value in successes)):
        reasons.append("actual_action_successes_unverified")
    if isinstance(executed, list) and any(not isinstance(action, str) or re.fullmatch(
            r"(?:turn_left|turn_right|(?:move|open|select) [1-9][0-9]*)", action) is None for action in executed):
        reasons.append("invalid_public_action_syntax")
    if (type(record.get("process_id")) is not int or type(record.get("construction_process_id")) is not int
            or record["process_id"] == record["construction_process_id"] or record.get("snapshot_restored") is not True):
        reasons.append("fresh_process_snapshot_replay_unverified")
    distance = record.get("selected_distance_m")
    distance_ok = type(distance) in (int, float) and math.isfinite(distance) and 0 <= distance <= 1.5
    pixels = record.get("selected_mask_pixels")
    if (record.get("selected_name") != target_name or not distance_ok or type(pixels) is not int or pixels <= 0
            or record.get("success") is not True or record.get("done") is not True):
        reasons.append("actual_terminal_select_success_unverified")
    return {"passed": not reasons, "reasons": reasons, "target_name": target_name,
            "executed_action_count": len(executed) if isinstance(executed, list) else None,
            "distance_method": DISTANCE_METHOD, "replay": copy.deepcopy(record)}


def classify_pair(kind, *, target_w0, target_w1, prefix, evidence, physical_w0, physical_w1,
                  single_edit, witness_w0, witness_w1, restore_deterministic,
                  anchor_pixels_w0, anchor_pixels_w1, common_observation_action=None, restore_max_abs_diff=None):
    """D1 input gate plus state restoration; native RGB difference is diagnostic."""
    if kind not in ("look_flip", "look_stay", "direct_flip", "direct_stay"):
        raise ValueError("Unknown T01v2 class")
    if restore_deterministic is not None and type(restore_deterministic) is not bool:
        raise ValueError("restore_deterministic must preserve bool or unknown")
    reasons = []
    for label, value in (("target_w0", target_w0.get("passed")), ("target_w1", target_w1.get("passed")),
                         ("physical_w0", physical_w0.get("valid_physical")), ("physical_w1", physical_w1.get("valid_physical")),
                         ("single_anchor_edit", single_edit.get("passed")), ("witness_w0", witness_w0.get("passed")),
                         ("witness_w1", witness_w1.get("passed")), ("candidate_prefix_visibility", prefix.get("candidates_visible_and_marked")),
                         ("same_process_fixed_agent_construction", prefix.get("construction_context_verified")),
                         ("state_restore", restore_deterministic),
                         ("evidence", evidence.get("evidence_passed")), ("policy_input_evidence", evidence.get("marked_evidence_passed"))):
        if value is not True:
            reasons.append(label)
    changed = target_w0.get("target_name") != target_w1.get("target_name")
    if changed != kind.endswith("flip"):
        reasons.append("target_change_disagrees_with_class")
    for target, witness in ((target_w0, witness_w0), (target_w1, witness_w1)):
        if witness.get("target_name") != target.get("target_name"):
            reasons.append("witness_target_does_not_match_resolved_target")
    if target_w0.get("anchor_name") != target_w1.get("anchor_name"):
        reasons.append("anchor_identity_changed")
    look = kind.startswith("look")
    if look:
        if prefix.get("same_input_at_t") is not True:
            reasons.append("look_input_not_actually_shared")
        h, w = evidence["resolution"]
        if not all(area_passes(value, h, w) for value in (anchor_pixels_w0, anchor_pixels_w1)):
            reasons.append("anchor_not_visible_after_one_observation")
        if not isinstance(common_observation_action, str) or not (common_observation_action in ("turn_left", "turn_right") or common_observation_action.startswith("move ")):
            reasons.append("common_first_observation_missing")
        for witness in (witness_w0, witness_w1):
            actions = witness.get("replay", {}).get("executed_actions") or []
            if not actions or actions[0] != common_observation_action:
                reasons.append("replay_does_not_start_with_common_observation")
    else:
        if prefix.get("native_public_prefix_exact") is not False or prefix.get("shared_prefix_verified") is True:
            reasons.append("direct_input_not_distinct")
        if any(type(value) is not int or value <= 0 for value in (anchor_pixels_w0, anchor_pixels_w1)):
            reasons.append("anchor_not_visible_at_t0")
    candidates = target_w0.get("candidate_names")
    separable = (isinstance(candidates, list) and len(candidates) == len(set(candidates)) == 2
                 and set(candidates) == set(target_w1.get("candidate_names") or [])
                 and all(target.get("target_name") in candidates for target in (target_w0, target_w1)))
    if not separable:
        reasons.append("select_target_identities_not_separable")
    d1_passed = not reasons
    return {"kind": kind, "acceptance_passed": d1_passed, "d1_candidate_passed": d1_passed,
            "reasons": sorted(set(reasons)),
            "same_input_at_t": prefix.get("same_input_at_t") if look else
                (prefix.get("native_public_prefix_exact") is True or prefix.get("shared_prefix_verified") is True),
            "targets_separable": separable, "restore_deterministic": restore_deterministic,
            "restore_max_abs_diff": restore_max_abs_diff, "restore_rgb_is_diagnostic_not_acceptance_gate": True,
            "acceptance_decision_source": POLICY["restore_acceptance_decision"]}

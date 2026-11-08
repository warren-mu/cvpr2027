"""Versioned T01v2 supplement-2 evidence; frozen earlier policies stay intact.

Look evidence uses only X's actual mask union; seeing X in both worlds is not
required. Direct pairs retain ordinary relevant-region evidence and require X
to occupy at least 250 pixels in each start view. All other gates are retained.
"""
import copy

from . import pair_v2_policy as original

EVIDENCE_VERSION = "anchor_union_look_v1"
EVIDENCE_POLICY = {
    "version": EVIDENCE_VERSION,
    "look_region": "union of the edited anchor X masks in the two first-observation frames",
    "look_per_world_anchor_minimum": None,
    "direct_per_world_anchor_pixels_at_255": 250,
    "evidence_channel_difference_exclusive": 10,
    "evidence_pixels_at_255": 250,
    "marked_input_evidence_still_required": True,
    "source": "tasks/T01v2.md supplement 2 (1f1359e)",
}
_ORIGINAL_CLASSIFY = original.classify_pair
_ORIGINAL_COMPARE = original.compare_view


def look_evidence(first, second, anchor_name):
    """Measure raw and marked RGB on X alone, including an absent per-world mask."""
    if not isinstance(anchor_name, str) or not anchor_name:
        raise ValueError("Look evidence requires the explicit stable anchor name")
    measured = _ORIGINAL_COMPARE(first, second, [anchor_name])
    measured.update(evidence_version=EVIDENCE_VERSION, anchor_name=anchor_name,
        anchor_pixels_w0=int(original._mask(first, anchor_name).sum()),
        anchor_pixels_w1=int(original._mask(second, anchor_name).sum()),
        relevant_names=[anchor_name], mask_union_scope="edited_anchor_only")
    return measured


def _verified_look_fields(evidence, anchor_name, counts):
    h, w = evidence["resolution"]
    if (evidence.get("evidence_version") != EVIDENCE_VERSION
            or evidence.get("anchor_name") != anchor_name
            or evidence.get("relevant_names") != [anchor_name]
            or evidence.get("mask_union_scope") != "edited_anchor_only"
            or evidence.get("evidence_channel_difference_exclusive") != 10
            or [evidence.get("anchor_pixels_w0"), evidence.get("anchor_pixels_w1")] != list(counts)):
        return False
    union = evidence.get("relevant_mask_union_pixels")
    if type(union) is not int or not max(counts) <= union <= min(h * w, sum(counts)):
        return False
    for count_key, pass_key in (("evidence_pixels", "evidence_passed"),
                                ("marked_evidence_pixels", "marked_evidence_passed")):
        count = evidence.get(count_key)
        if type(count) is not int or not 0 <= count <= union:
            return False
        if evidence.get(pass_key) is not original.area_passes(count, h, w):
            return False
    return True


def classify_pair(kind, *, target_w0, target_w1, prefix, evidence, physical_w0, physical_w1,
                  single_edit, witness_w0, witness_w1, restore_deterministic,
                  anchor_pixels_w0, anchor_pixels_w1, common_observation_action=None, restore_max_abs_diff=None):
    """Replace exactly the superseded visibility gate; never synthesize counts."""
    result = _ORIGINAL_CLASSIFY(kind, target_w0=target_w0, target_w1=target_w1, prefix=prefix,
        evidence=evidence, physical_w0=physical_w0, physical_w1=physical_w1, single_edit=single_edit,
        witness_w0=witness_w0, witness_w1=witness_w1, restore_deterministic=restore_deterministic,
        anchor_pixels_w0=anchor_pixels_w0, anchor_pixels_w1=anchor_pixels_w1,
        common_observation_action=common_observation_action, restore_max_abs_diff=restore_max_abs_diff)
    reasons = set(result["reasons"])
    counts = (anchor_pixels_w0, anchor_pixels_w1)
    h, w = evidence["resolution"]
    if kind.startswith("look"):
        reasons.discard("anchor_not_visible_after_one_observation")
        if not _verified_look_fields(evidence, target_w0.get("anchor_name"), counts):
            reasons.add("look_anchor_union_evidence_unverified")
    elif not all(original.area_passes(count, h, w) for count in counts):
        reasons.add("anchor_visibility_below_250_at_t0")
    result.update(reasons=sorted(reasons), acceptance_passed=not reasons, d1_candidate_passed=not reasons,
                  evidence_version=EVIDENCE_VERSION,
                  evidence_rule="first-observation anchor union" if kind.startswith("look") else "direct per-world anchor visibility")
    return result


def choose_look(sim, worlds, targets, identities, reachable, prefix):
    """Actual common first action, then measured select or verified inverse/select.

Both candidates are selectable at v0, so a verified inverse supplies a bounded
continuation without requiring X to be visible in both first-observation views.
All actions are still executed separately in both worlds and replayed freshly.
"""
    from . import pair_v2_shard as shard
    actions, unavailable = shard.observation_actions(sim, reachable, worlds[0])
    choices, attempts = [], []
    for action in actions:
        observed, plans, details = [], [], []
        try:
            for world, target in zip(worlds, targets):
                _, restoration = sim.restore_snapshot(world)
                if restoration.get("restore_deterministic") is not True:
                    raise ValueError("Planner snapshot restoration failed the state-level gate")
                world_action = copy.deepcopy(action)
                if action["kind"] == "move":
                    world_action["path"] = sim.shortest_path(world["agent"]["position"], action["position"])
                    if world_action["path"].get("complete") is not True:
                        raise ValueError("Common public move is not feasible in this restored world")
                observation = shard.marked(sim, sim.observe(world_action), identities)
                observed.append(observation)
                possible, label, selection = shard.eligible_select(sim, observation, target)
                plan = [world_action]
                if not possible:
                    reverse = shard.inverse_action(sim, action, world)
                    returned = shard.marked(sim, sim.observe(reverse), identities)
                    possible, label, selection = shard.eligible_select(sim, returned, target)
                    plan.append(reverse)
                if not possible:
                    raise ValueError("Inverse observation did not reach a valid select")
                plan.append({"kind": "select", "label": label})
                plans.append(plan)
                details.append(selection)
            evidence = look_evidence(observed[0], observed[1], identities["anchor_name"])
            counts = [evidence["anchor_pixels_w0"], evidence["anchor_pixels_w1"]]
            passed = evidence["evidence_passed"] and evidence["marked_evidence_passed"]
            attempts.append({"action": shard.action_text(action), "evidence": evidence,
                             "anchor_pixels": counts, "passed": passed,
                             "plan_lengths": [len(plan) for plan in plans]})
            if passed:
                choices.append({"action": action, "after": observed, "plans": plans,
                    "evidence": evidence, "anchor_pixels": counts,
                    "planning": {"method": "shortest measured continuation after a common one-action evidence acquisition",
                        "action_counts": [len(plan) for plan in plans], "selection_measurements": details,
                        "lower_bound_including_evidence_and_select": 2}})
                if all(len(plan) == 2 for plan in plans):
                    break
        except (TimeoutError, KeyboardInterrupt, SystemExit):
            raise
        except Exception as error:
            attempts.append({"action": shard.action_text(action), "passed": False,
                             "error": f"{type(error).__name__}: {error}"})
    if not choices:
        return None, {"actions": attempts, "unavailable_moves": unavailable}
    choice = min(choices, key=lambda row: (sum(map(len, row["plans"])), max(map(len, row["plans"])),
                                         shard.action_text(row["action"])))
    choice["planning"].update(first_action_search=attempts, unavailable_moves=unavailable,
                             global_two_action_lower_bound_attained=all(len(plan) == 2 for plan in choice["plans"]))
    return choice, {"actions": attempts, "unavailable_moves": unavailable}

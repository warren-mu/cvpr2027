"""Human-approved 2026-11-08 T01 criteria; pure CPU comparisons only.

The 2 cm bound is the error from requested pose after settling, not the edit
travel distance. This module does not relax exact public-prefix equality.
"""
import math

POLICY = {
    "id": "t01_human_20261108",
    "edited_settling_position_error_max_m": 0.02,
    "other_object_displacement_max_m": 0.01,
    "evidence_channel_difference_exclusive": 10,
    "evidence_min_pixels_at_255": 250,
    "evidence_reference_resolution": 255,
    "edit_travel_distance_max_m": None,
    "prefix_equality": "exact RGB, marked RGB, public text and history",
    "source": "docs/METRICS_SPEC.md section 1; human instruction 2026-11-08",
}


def evidence_passes(count, height, width):
    if any(type(x) is not int or x < 0 for x in (count, height, width)) or not height or not width:
        raise ValueError("Require nonnegative integer count and positive image dimensions")
    if count > height * width:
        raise ValueError("Evidence count exceeds image area")
    # Integer cross multiplication avoids rounding the scientific boundary.
    return count * 255 * 255 >= 250 * height * width


def position_distance(first, second):
    values = [float(first[k]) - float(second[k]) for k in ("x", "y", "z")]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Nonfinite object position")
    return math.sqrt(sum(value * value for value in values))


def rotation_delta_degrees(first, second):
    """Record wrapped Euler component changes; not a geodesic rotation metric."""
    return {k: abs((float(first[k]) - float(second[k]) + 180) % 360 - 180) for k in ("x", "y", "z")}


def indexed_objects(objects):
    indexed = {row["name"]: row for row in objects}
    if len(indexed) != len(objects):
        raise ValueError("Duplicate stable object name")
    return indexed


def measure_settled_poses(baseline, requested_pose, realized, edited_name):
    before, after = indexed_objects(baseline), indexed_objects(realized)
    if requested_pose.get("objectName") != edited_name or edited_name not in before or edited_name not in after:
        raise ValueError("Edited object/request identity is missing or inconsistent")
    exact_inventory = set(before) == set(after)
    deltas = {name: {
        "position_delta_m": position_distance(before[name]["position"], after[name]["position"]),
        "rotation_component_delta_degrees": rotation_delta_degrees(before[name]["rotation"], after[name]["rotation"]),
        "before_position": before[name]["position"], "after_position": after[name]["position"],
        "before_rotation": before[name]["rotation"], "after_rotation": after[name]["rotation"],
    } for name in sorted(set(before) & set(after))}
    error = position_distance(requested_pose["position"], after[edited_name]["position"])
    other = {name: row for name, row in deltas.items() if name != edited_name}
    return {
        "inventory_exact": exact_inventory,
        "missing": sorted(set(before) - set(after)), "added": sorted(set(after) - set(before)),
        "edited_settling_position_error_m": error,
        "edited_rotation_error_components_degrees": rotation_delta_degrees(requested_pose["rotation"], after[edited_name]["rotation"]),
        "edited_travel_distance_m": deltas[edited_name]["position_delta_m"],
        "all_object_pose_deltas": deltas,
        "maximum_other_object_displacement_m": max((r["position_delta_m"] for r in other.values()), default=0.0),
        "position_bounds_passed": exact_inventory and error <= 0.02 and all(r["position_delta_m"] <= 0.01 for r in other.values()),
        "rotation_is_recorded_without_an_unapproved_acceptance_threshold": True,
    }


def compare_inputs(first, second, relevant_ids):
    import numpy as np
    from .pair_simulator import compare_inputs as original_compare
    result = original_compare(first, second, relevant_ids)
    delta = np.abs(first["rgb"].astype(np.int16) - second["rgb"].astype(np.int16))
    union = np.zeros(delta.shape[:2], dtype=bool)
    for obs in (first, second):
        for identity in relevant_ids:
            if identity in obs["masks"]:
                mask = np.asarray(obs["masks"][identity])
                if mask.shape != union.shape or mask.dtype != np.bool_:
                    raise ValueError("Relevant instance mask shape or dtype mismatch")
                union |= mask
    result["relevant_nonzero_difference_pixels"] = result["relevant_difference_pixels"]
    count = int((np.any(delta > 10, axis=2) & union).sum())
    result.update(relevant_difference_pixels=count, evidence_channel_difference_exclusive=10,
                  evidence_passed=evidence_passes(count, *delta.shape[:2]),
                  evidence_resolution=list(delta.shape[:2]),
                  minimum_evidence_pixels=math.ceil(250 * delta.shape[0] * delta.shape[1] / (255 * 255)),
                  acceptance_policy=POLICY["id"])
    return result

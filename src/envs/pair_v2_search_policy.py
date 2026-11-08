"""Pure, versioned T01v2 search ordering; never a physical validity certificate.

Category spelling follows the pinned native SimObjType enum. Sliced/Cracked
are native suffixes; Broken is an explicitly excluded forward-compatible
suffix, not a claim that the pinned enum contains such a category. Native
breakage can also preserve objectType, so isBroken/isSliced are checked for
objects we would move. Existing unmoved candidates are deliberately exempt.
"""
import copy
import json

from src.envs.pair_v2_construct import identity_sets

SEARCH_VERSION = "natural_union_clone_search_v1"
SEARCH_POLICY = {
    "excluded_moved_object_types": ["Apple", "Tomato", "Potato", "Egg", "Lettuce",
                                    "BasketBall", "BaseballBat", "Pen", "Pencil"],
    "excluded_native_variant_types": ["AppleSliced", "TomatoSliced", "BreadSliced",
                                      "PotatoSliced", "LettuceSliced", "EggCracked", "EggShell"],
    "excluded_type_suffixes": ["Sliced", "Cracked", "Broken"],
    "category_matching": "exact case-sensitive metadata.objectType; suffixes apply to every category",
    "excluded_state_flags": ["isBroken", "isSliced"],
    "exclusion_scope": "anchor and clone source only; natural unmoved candidates unrestricted",
    "native_category_source": "https://github.com/allenai/ai2thor/blob/f0825767cd50d69f666c7f282e54abfe58f1e917/unity/Assets/Scripts/SimObjType.cs",
    "native_state_source": "https://github.com/allenai/ai2thor/blob/f0825767cd50d69f666c7f282e54abfe58f1e917/unity/Assets/Scripts/BaseFPSAgentController.cs#L1481",
    "clone_source_order": ["objectType", "name"],
    "clone_source_rule": "exactly one object of its category, pickupable true, eligible to move",
    "maximum_clone_attempts": 6,
    "clone_controller_policy": "fresh Controller and original-scene preflight per source; stop at first success",
    "candidate_groups": "all natural groups union the successful clone group; no group count truncation",
    "combination_order": ["target_type", "anchor_type", "candidate_names", "anchor_name"],
    "combination_schedule": "round_robin_one_proposal_per_combination",
    "consecutive_physical_rejection_limit": 10,
    "physical_rejection_streak_reset": "any completed attempt without a physical rejection",
}


def excluded_clone_source_type(object_type):
    if not isinstance(object_type, str) or not object_type:
        raise ValueError("A native objectType string is required")
    return (object_type in SEARCH_POLICY["excluded_moved_object_types"]
            or object_type in SEARCH_POLICY["excluded_native_variant_types"]
            or object_type.endswith(tuple(SEARCH_POLICY["excluded_type_suffixes"])))


def eligible_moved_object(obj):
    """Apply the same declared filter to an anchor or a proposed clone source."""
    return (obj.get("pickupable") is True and not excluded_clone_source_type(obj.get("objectType"))
            and not any(obj.get(flag) is True for flag in SEARCH_POLICY["excluded_state_flags"]))


def object_inventory(objects):
    result = {}
    for obj in objects:
        if not isinstance(obj, dict) or not isinstance(obj.get("name"), str) or not obj["name"]:
            raise ValueError("Each object needs a stable nonempty name")
        if obj["name"] in result:
            raise ValueError("Duplicate stable object name")
        excluded_clone_source_type(obj.get("objectType"))  # Validate category, not eligibility.
        result[obj["name"]] = obj
    return result


def eligible_clone_sources(objects):
    inventory = object_inventory(objects)
    groups = {}
    for obj in inventory.values():
        groups.setdefault(obj["objectType"], []).append(obj)
    candidates = [items[0] for items in groups.values()
                  if len(items) == 1 and eligible_moved_object(items[0])]
    return copy.deepcopy(sorted(candidates, key=lambda obj: (obj["objectType"], obj["name"]))
                         [:SEARCH_POLICY["maximum_clone_attempts"]])


def union_identity_sets(objects, lineage=None):
    """Enumerate actual complete categories, preserving natural rolling pairs.

Lineage must have been independently verified by the caller. This helper binds
its two actual names to the inventory, but does not certify creation/settling.
"""
    inventory = object_inventory(objects)
    clone_names = None
    if lineage is not None:
        clone_names = lineage.get("candidate_names")
        if (not isinstance(clone_names, list) or len(clone_names) != 2
                or len(set(clone_names)) != 2 or not set(clone_names) <= set(inventory)
                or len({inventory[name]["objectType"] for name in clone_names}) != 1):
            raise ValueError("Clone lineage is not a distinct same-category pair in this inventory")
        clone_names = set(clone_names)
    result = []
    for group in identity_sets(list(inventory.values())):
        if not eligible_moved_object(inventory[group["anchor_name"]]):
            continue
        group["search_candidate_origin"] = ("cloned" if set(group["candidate_names"]) == clone_names else "natural")
        result.append(group)
    return sorted(result, key=lambda group: (group["target_type"], group["anchor_type"],
                                              group["candidate_names"], group["anchor_name"]))


def identity_key(group):
    names, anchor = group.get("candidate_names"), group.get("anchor_name")
    if (not isinstance(names, list) or len(names) != 2 or any(not isinstance(n, str) or not n for n in names)
            or len(set(names)) != 2 or not isinstance(anchor, str) or not anchor or anchor in names):
        raise ValueError("Combination requires two distinct candidates and a distinct anchor")
    return json.dumps({"candidate_names": sorted(names), "anchor_name": anchor}, sort_keys=True)


class RoundRobinScheduler:
    """Only ordering/streak state; proposal exhaustion is reported by the caller."""
    key = staticmethod(identity_key)

    def __init__(self, groups):
        self.groups = copy.deepcopy(list(groups))
        self.keys = [self.key(group) for group in self.groups]
        if len(set(self.keys)) != len(self.keys):
            raise ValueError("Duplicate candidate/anchor combination")
        self.states = {key: {"key": key, "attempts": 0, "physical_rejections": 0,
                            "consecutive_physical_rejections": 0, "skipped": False, "exhausted": False}
                       for key in self.keys}
        self.cursor = 0

    def _state(self, group_or_key):
        key = group_or_key if isinstance(group_or_key, str) else self.key(group_or_key)
        if key not in self.states:
            raise ValueError("Unknown search combination")
        return self.states[key]

    def next(self):
        for _ in self.groups:
            index = self.cursor
            self.cursor = (self.cursor + 1) % len(self.groups)
            state = self.states[self.keys[index]]
            if not state["skipped"] and not state["exhausted"]:
                return copy.deepcopy(self.groups[index])
        return None

    def observe(self, group_or_key, *, physical_rejected):
        if type(physical_rejected) is not bool:
            raise ValueError("An observed physical-rejection boolean is required")
        state = self._state(group_or_key)
        if state["skipped"] or state["exhausted"]:
            raise ValueError("Cannot observe a retired combination")
        state["attempts"] += 1
        state["physical_rejections"] += int(physical_rejected)
        state["consecutive_physical_rejections"] = state["consecutive_physical_rejections"] + 1 if physical_rejected else 0
        state["skipped"] = state["consecutive_physical_rejections"] >= SEARCH_POLICY["consecutive_physical_rejection_limit"]
        return copy.deepcopy(state)

    def exhaust(self, group_or_key):
        self._state(group_or_key)["exhausted"] = True

    def snapshot(self):
        return [copy.deepcopy(self.states[key]) for key in self.keys]
